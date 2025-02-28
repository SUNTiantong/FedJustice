# Python version: 3.11
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchcontrib.optim import SWA
from component.withadv import Run_adv_model,add_laplace_noise
from torch.amp import autocast, GradScaler
from component.plot_func import plot_loss,plot_accuracy
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 定义函数计算概率
def compute_prob(y_pred, y_true, s, target_s, target_y):
    # 筛选条件 s = target_s, Y = target_y
    indices = [i for i in range(len(y_true)) if s[i] == target_s and y_true[i] == target_y]
    if not indices:
        return 0.0
    # 预测为 1 的数量 / 满足条件的总数量
    pred_1_count = sum(y_pred[i] == 1 for i in indices)
    return pred_1_count / len(indices)
    #s=sensitive_features
def DEO(s,y_true,y_pred):
    M_deo = 0
    # 替换 'male' 为 1，'female' 为 0
    # s = [1 if gender == "male" else 0 for gender in s]
    sensitive_values=list(set(s))
    for y in [0, 1]:
        prob_s0 = compute_prob(y_pred, y_true, s, target_s=sensitive_values[0], target_y=y)
        prob_s1 = compute_prob(y_pred, y_true, s, target_s=sensitive_values[1], target_y=y)
        M_deo += abs(prob_s0 - prob_s1)
    return M_deo
    

def compute_probdpd(y_pred, s, target_s):
    # 筛选条件 s = target_s, Y = target_y
    indices = [i for i in range(len(y_pred)) if s[i] == target_s]
    if not indices:
        return 0.0
    # 预测为 1 的数量 / 满足条件的总数量
    pred_1_count = sum(y_pred[i] == 1 for i in indices)
    return pred_1_count / len(indices)
    #s=sensitive_features    
def DPD(s,y_pred):
    M_dpd = 0
    # 替换 'male' 为 1，'female' 为 0
    # s = [1 if gender == "male" else 0 for gender in s]
    sensitive_values=list(set(s))
    prob_s0 = compute_probdpd(y_pred, s, target_s=sensitive_values[0])
    prob_s1 = compute_probdpd(y_pred, s, target_s=sensitive_values[1])
    M_dpd += abs(prob_s0 - prob_s1)
    return M_dpd

class LocalUpdate(object):
    def __init__(self, args,tokenizer, gender_dataset,dataset):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.g=gender_dataset
        self.B_S=args.batch_size#self.args.batch_size
        # self.B_S=50
        self.gender_dataset = DataLoader(gender_dataset, batch_size=self.B_S,num_workers=4, shuffle=True, pin_memory=True)
        self.dataset= DataLoader(dataset, batch_size=self.B_S, num_workers=4,shuffle=True, pin_memory=True)

        self.tokenizer= tokenizer
        self.length= len(gender_dataset)

        # 设置填充标记为结束标记
        tokenizer.pad_token = tokenizer.eos_token


    M_deo=0
    M_dpd=0
    def train(self, lora_model,adv_model,client_idx,round,server_model):
        lora_model = lora_model.to("cuda:0")
        
        # lora_model=lora_model.module
        # adv_model = CategoricalEmb(lora_model.get_input_embeddings())#Embedding(50257, 2048)
        emb=lora_model.get_input_embeddings()
        #输入：模型的输入通常是一系列的 token ID，这些 ID 是通过 tokenizer 处理文本数据得到的。例如，如果我们输入一句话，tokenizer 会将其转换成一系列的数字 ID。
        #输出：每个 token ID 会被 Embedding 层转换成一个固定维度的嵌入向量。在这个例子中，每个 token 会被转换为一个 2048 维的向量。
        optimizer = torch.optim.AdamW(lora_model.parameters(), lr=5e-4)#,weight_decay=1e-5
        adv_optimiser = SWA(torch.optim.Adam(adv_model.parameters(), lr=1e-1), swa_start=10, swa_freq=5, swa_lr=0.05)
        loss_fn = nn.CrossEntropyLoss()


        optimizer.zero_grad()
        #lora_model.get_input_embeddings().weight
        #lora_model.get_input_embeddings().weight.requires_grad
        # train and update
        accuracy = 0
        total_loss = 0 
        
        #普通样本训练
        y_pred_deo=[]
        # nnn=0
        # self.args.local_epochs=10  #########################
        M_deo=0
        epoch_loss = []
        epoch_accuracy=[]
        for epoch in range(self.args.local_epochs):
            lora_model.train()
            adv_model.eval()
            batch_loss=[]
            now_batch_accuracy_list=[]
            for batch_idx, data in enumerate(self.gender_dataset):  # 遍历数据集
                prompts = list(data[0][0])  # 提取 prompt 列表
                label = data[2]
                label_list=label.tolist()
                label_list_str = [str(x.item()) for x in label ]#data[2] = tensor([0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0])
                # 混合精度训练
                with autocast(device_type="cuda", enabled=True):  
                    # Tokenize inputs and prepare for batch processing
                    model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
                    model_inputs.input_ids = model_inputs.input_ids.clamp(0, lora_model.config.vocab_size - 1)#50257-1

                    model_inputs.input_ids=model_inputs.input_ids.to("cuda:0")
                    model_inputs.attention_mask = model_inputs.attention_mask.to("cuda:0")
                    output_ids= self.tokenizer(label_list_str, return_tensors="pt", padding=True)                
                    output_ids=output_ids.to("cuda:0")
                    
#self.tokenizer.decode([15], skip_special_tokens=True) = '<|endoftext|>'
# self.tokenizer('0', return_tensors="pt",padding=True)
# {'input_ids': tensor([[15]]), 'attention_mask': tensor([[1]])}
# self.tokenizer('1', return_tensors="pt",padding=True)
# {'input_ids': tensor([[16]]), 'attention_mask': tensor([[1]])} 
                    if batch_idx < self.args.run_adv * len(self.gender_dataset):
                        embeddings, attention_mask = adv_model(model_inputs.input_ids, model_inputs.attention_mask)
                        logits = lora_model(inputs_embeds=embeddings, attention_mask=attention_mask).logits[:, -1, :].squeeze(1)                        
                    elif batch_idx >= self.args.run_adv * len(self.gender_dataset):
                        
                        # 设置 Laplace 噪声的尺度
                        noise_scale = 0  # 调整噪声的强度0.5
                        if noise_scale >0:
                            # 调用函数，添加 Laplace 噪声
                            embedding_with_noise = add_laplace_noise(model_inputs.input_ids.long(), noise_scale)
                        else:
                            embedding_with_noise = model_inputs.input_ids.long()                        
                        
                        logits = lora_model(embedding_with_noise, model_inputs.attention_mask).logits[:, -1, :].squeeze(1)
                    else:
                        print( "self.args.run_adv设置出错了")
                        raise ValueError
                    encoded_tensor=torch.argmax(logits, dim=-1)#self.tokenizer.decode([15], skip_special_tokens=True)
                    # predict_list = [self.tokenizer.decode([token_id]) for token_id in encoded_tensor.tolist()]
                    predict_list = list(map(lambda token_id: int(self.tokenizer.decode([token_id])) if self.tokenizer.decode([token_id]).isdigit() else self.tokenizer.decode([token_id]), encoded_tensor.tolist()))
                    
                    y_pred_deo+=predict_list# [1,0,1,' A'，0]
                    output_ids= output_ids.to(logits.device)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), output_ids.input_ids.view(-1))
                    # FedProx 近端项参数
                    # mu = self.args.mu if hasattr(self.args, 'mu') else 0.1
                    # 计算 FedProx 近端项
                    if self.args.run_adv == -2:
                        mu=1
                        proximal_term = 0.0
                        for w, w_t in zip(lora_model.parameters(), server_model.to("cuda:0").parameters()):
                            proximal_term += (w - w_t).norm(2)
                        loss += (mu / 2) * proximal_term
                    loss.backward()
                    del logits
                    torch.cuda.empty_cache()
                # torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)

                with torch.no_grad():
                    # print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}")
                    # 统计 loss 和 accuracy
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_loss.append(loss.item())###############
                    now_batch_accuracy= torch.sum(encoded_tensor == output_ids.input_ids.view(-1)).item()
                    now_batch_accuracy_list.append(now_batch_accuracy/ self.B_S)
                    accuracy +=now_batch_accuracy
                    # print("round:",round,"client_idx:",client_idx,"at epoch ",epoch,"   ","now_batch_accuracy", now_batch_accuracy/ self.B_S,'   ',"now_batch_loss:",loss.item())############

            accuracy=accuracy / self.length
            epoch_loss.append(sum(batch_loss))
            epoch_accuracy.append(accuracy)
            print("round:",round,"client_idx:",client_idx,"finish epoch ",epoch,"   ","epoch accuracy",accuracy ,'   ',"epoch_loss:",sum(batch_loss),'\n')############

            
            adv_model.train()     
            lora_model.eval()
            if self.args.run_adv > 0:
                Run_adv_model(lora_model,adv_model,adv_optimiser,dataset=self.dataset,tokenizer=self.tokenizer)
            else: continue
        file_path_loss = "/home/chen/pyh/FedJudge-main/Result/round{}_client_idx{}_epoch_loss_plot.png".format(round,client_idx)
        file_path_accuracy = "/home/chen/pyh/FedJudge-main/Result/round{}_client_idx{}_epoch_accuracy_plot.png".format(round,client_idx)
        # plot_loss(range(1, epoch + 2), epoch_loss,file_path_loss)  # 绘制 Loss 曲线
        # plot_accuracy(range(1, epoch + 2), epoch_accuracy,file_path_accuracy)  # 绘制 Accuracy 曲线
            

        return lora_model.state_dict(), sum(epoch_loss) / len(epoch_loss), M_deo



def test(server_model,test_dataset,batch_size,tokenizer):
    # Set model to evaluation mode
    lora_model=server_model
    lora_model.eval()
    lora_model = lora_model.to("cuda:0")
    test_dataset=test_dataset
    test_dataset_loader=DataLoader(test_dataset, batch_size=batch_size,num_workers=4, shuffle=True, pin_memory=True)

    test_loss = 0
    correct = 0
    total_samples =len(test_dataset)
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=5e-4)#,weight_decay=1e-5
    loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    accuracy = 0
    y_pred_deo=[]
    batch_loss=[]

    for batch_idx, data in enumerate(test_dataset_loader):  # 遍历数据集
        prompts = list(data[0][0])  # 提取 prompt 列表
        label = data[2]

        label_list_str = [str(x.item()) for x in label ]#data[2] = tensor([0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0])

        with torch.no_grad():
            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            model_inputs.input_ids = model_inputs.input_ids.clamp(0, lora_model.config.vocab_size - 1)#50257-1
            model_inputs.input_ids=model_inputs.input_ids.long().to("cuda:0")
            model_inputs.attention_mask = model_inputs.attention_mask.to("cuda:0")
# model_inputs.input_ids.shape
# torch.Size([50, 88])
            output_ids= tokenizer(label_list_str, return_tensors="pt", padding=True)                
            output_ids=output_ids.to("cuda:0")
          #GPU 0 has a total capacity of 47.53 GiB of which 2.00 MiB is free. Including non-PyTorch memory, this process has 47.51 GiB memory in use.   
            logits = lora_model(model_inputs.input_ids, model_inputs.attention_mask).logits[:, -1, :].squeeze(1)
            encoded_tensor=torch.argmax(logits, dim=-1)
            predict_list = list(map(lambda token_id: int(tokenizer.decode([token_id])) if tokenizer.decode([token_id]).isdigit() else tokenizer.decode([token_id]), encoded_tensor.tolist()))
            y_pred_deo+=predict_list# [1,0,1,' A'，0]
            output_ids= output_ids.to(logits.device)

            # 统计 loss 和 accuracy
            loss = loss_fn(logits.view(-1, logits.size(-1)), output_ids.input_ids.view(-1))
            batch_loss.append(loss.item())   
            accuracy +=torch.sum(encoded_tensor == output_ids.input_ids.view(-1)).item()
            del logits
            torch.cuda.empty_cache()            
    test_accuracy=accuracy / total_samples
    avg_loss=sum(batch_loss)/total_samples
    test_avg_loss=avg_loss
    print("test accuracy",test_accuracy ,'   ',"test avg_loss:",test_avg_loss)

    
    #计算公平性
    # 将 dataset 中的每个特征分离成单独的向量
    gender_deo=[item[1] for item in test_dataset]    # 抽取 gender 列
    label_deo = [item[2] for item in test_dataset]    # 抽取 label 列
    y_pred_deo = y_pred_deo  # Predicted label (the class with the highest logit score)

    from collections import Counter
    # 统计每个元素出现的次数
    counter = Counter(gender_deo)
    # 获取出现次数最多的元素
    most_common_element, most_common_count = counter.most_common(1)[0]
    # 修改 gender_deo 列表，给非 most_common_element 的元素加上前缀 "Isnot"
    modified_gender_deo = [f"Isnot{most_common_element}" if element != most_common_element else element for element in gender_deo]

    M_deo = DEO(s=modified_gender_deo,y_true=label_deo,y_pred=y_pred_deo)     
    M_dpd = DPD(s=modified_gender_deo,y_pred=y_pred_deo)
    print("M_DEO:",M_deo,'\n')   
    print("M_DPD:",M_dpd,'\n')   

    y_pred_deo=[]  

    return test_accuracy,test_avg_loss,M_deo,M_dpd



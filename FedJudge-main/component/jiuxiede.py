#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from transformers import AutoTokenizer
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from torchcontrib.optim import SWA
import torchvision
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, LogitsProcessor
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def make_gumbel_softmax(emb):

    class GumbelSoftmax(torch.autograd.Function):
        E = emb

        @staticmethod
        def forward(ctx, input):
            L, N = input.shape
            with torch.enable_grad():
                softmaxed = F.gumbel_softmax(input, dim = 1)
            output  = torch.argmax(softmaxed, dim = 1)
            ctx.save_for_backward(input, softmaxed)
            return output, GumbelSoftmax.E(output)

        @staticmethod
        def backward(ctx, temp,grad_output):
            inp, softmaxed = ctx.saved_tensors
            grad_input = torch.autograd.grad(softmaxed, inp, grad_outputs=torch.mm(grad_output,GumbelSoftmax.E.weight.T))
            return grad_input
    
    return GumbelSoftmax


class CategoricalEmb(nn.Module):
    def __init__(self,emb):
        super(CategoricalEmb,self).__init__()
        fair_sent_dist = torch.randn(10,emb.weight.data.shape[0]).to("cuda:0")
        self.register_parameter("fair_sent_dist",nn.Parameter(fair_sent_dist))
        self.f_gumble_softmax = make_gumbel_softmax(emb)
        self.embeddings = emb
        self.embeddings.weight.requires_grad = False
        self.embeddings.to("cuda:0")

    def forward(self,input_ids,attn_mask):
        fairprompt_ids,fair_prompt = self.f_gumble_softmax.apply(self.fair_sent_dist)
        fair_prompt  = fair_prompt.repeat(input_ids.shape[0],1,1)
        fairprompt_ids = fairprompt_ids.repeat(input_ids.shape[0],1)
        embeddings = self.embeddings(input_ids)
        embeddings = torch.cat([fair_prompt,embeddings],dim=1)

        attention_mask = torch.cat([torch.ones_like(fairprompt_ids),attn_mask],dim=1)

        return embeddings,attention_mask

class RestrictToBinaryLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.allowed_tokens = [tokenizer.encode("0")[0], tokenizer.encode("1")[0]]  # 获取 0 和 1 的 token IDs

    def __call__(self, input_ids, scores):
        restricted_scores = torch.full_like(scores, -float("inf"))  # 初始化所有 token 的分数为负无穷
        restricted_scores[:, self.allowed_tokens] = scores[:, self.allowed_tokens]  # 仅保留 0 和 1 的分数
        return restricted_scores
# Modify logits processing and model output
# Ensure logits are processed correctly and outputs are valid.
def get_classification(logits,tokenizer):
    # Applying the RestrictToBinaryLogitsProcessor
    logits_processor = LogitsProcessorList([RestrictToBinaryLogitsProcessor(tokenizer)])
    processed_logits = logits_processor(input_ids=None, scores=logits)
    
    # Ensure only one of the allowed tokens (0 or 1) is selected
    predicted_token_ids = torch.argmax(processed_logits, dim=-1)  # Predict the token ID for 0 or 1
    return predicted_token_ids

def reinitialize_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):  # 检查层是否有 `reset_parameters` 方法
            layer.reset_parameters()
# 禁用 Beta transforms 的警告
torchvision.disable_beta_transforms_warning()




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
    for y in [0, 1]:
        prob_s0 = compute_prob(y_pred, y_true, s, target_s="Female", target_y=y)
        prob_s1 = compute_prob(y_pred, y_true, s, target_s="Male", target_y=y)
        M_deo += abs(prob_s0 - prob_s1)
    return M_deo
    

class LocalUpdate(object):
    def __init__(self, args,tokenizer, gender_dataset,dataset):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.g=gender_dataset
        self.B_S=args.batch_size#self.args.batch_size
        self.B_S=256 
        self.gender_dataset = DataLoader(gender_dataset, batch_size=self.B_S,num_workers=4, shuffle=True)
        self.dataset= DataLoader(dataset, batch_size=self.B_S, num_workers=4,shuffle=True)
        self.tokenizer= tokenizer
        self.length= len(gender_dataset)

        # 设置填充标记为结束标记
        tokenizer.pad_token = tokenizer.eos_token


    M_deo=0
    def train(self, lora_model):
        lora_model = lora_model.to(self.args.device)
        adv_model = CategoricalEmb(lora_model.get_input_embeddings().to(self.args.device))#Embedding(50257, 2048)
        emb=lora_model.get_input_embeddings()
        #输入：模型的输入通常是一系列的 token ID，这些 ID 是通过 tokenizer 处理文本数据得到的。例如，如果我们输入一句话，tokenizer 会将其转换成一系列的数字 ID。
        #输出：每个 token ID 会被 Embedding 层转换成一个固定维度的嵌入向量。在这个例子中，每个 token 会被转换为一个 2048 维的向量。
        optimizer = torch.optim.AdamW(lora_model.parameters(), lr=5e-4)#,weight_decay=1e-5
        adv_optimiser = SWA(torch.optim.Adam(adv_model.parameters(), lr=1e-1), swa_start=10, swa_freq=5, swa_lr=0.05)
        loss_fn = nn.CrossEntropyLoss().to(self.args.device)


        lora_model.train()
        adv_model.eval()
        optimizer.zero_grad()
        #lora_model.get_input_embeddings().weight
        #lora_model.get_input_embeddings().weight.requires_grad
        # train and update
        accuracy = 0
        total_loss = 0 
        
        

        #普通样本训练
        y_pred_deo=[]
        nnn=0
        self.args.local_ep=1
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            
            for batch_idx, data in enumerate(self.gender_dataset):  # 遍历数据集
                prompts = list(data[0][0])  # 提取 prompt 列表
                label = data[2].to(self.args.device)
                label_list=label.tolist()
                label_list_str = [str(x.item()) for x in label ]#data[2] = tensor([0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0])
                
                # Tokenize inputs and prepare for batch processing
                model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
                model_inputs.input_ids = model_inputs.input_ids.clamp(0, lora_model.config.vocab_size - 1).to(self.args.device)#50257-1
                output_ids= self.tokenizer(label_list_str, return_tensors="pt", padding=True).to(self.args.device)
                
                model_inputs.attention_mask = model_inputs.attention_mask.to(self.args.device)
#self.tokenizer.decode([15], skip_special_tokens=True) = '<|endoftext|>'
# self.tokenizer('0', return_tensors="pt",padding=True)
# {'input_ids': tensor([[15]]), 'attention_mask': tensor([[1]])}
# self.tokenizer('1', return_tensors="pt",padding=True)
# {'input_ids': tensor([[16]]), 'attention_mask': tensor([[1]])}
                # 获取输入的嵌入向量
                embeddings, attention_mask = adv_model(model_inputs.input_ids, model_inputs.attention_mask)

                logits = lora_model(inputs_embeds=embeddings, attention_mask=attention_mask).logits[:, -1, :].squeeze(1)
                encoded_tensor=torch.argmax(logits, dim=-1)#self.tokenizer.decode([15], skip_special_tokens=True)
                # predict_list = [self.tokenizer.decode([token_id]) for token_id in encoded_tensor.tolist()]
                predict_list = list(map(lambda token_id: int(self.tokenizer.decode([token_id])) if self.tokenizer.decode([token_id]).isdigit() else self.tokenizer.decode([token_id]), encoded_tensor.tolist()))
                y_pred_deo+=predict_list# [1,0,1,' A'，0]
                loss = loss_fn(logits.view(-1, logits.size(-1)), output_ids.input_ids.view(-1))
                # 反向传播与优化
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    # print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}")
                    # 统计 loss 和 accuracy
                    batch_loss.append(loss.item())
                    epoch_loss.append(sum(batch_loss)/len(batch_loss))
                    now_batch_accuracy= torch.sum(encoded_tensor == output_ids.input_ids.view(-1)).item()
                    accuracy +=now_batch_accuracy
                    print("now_batch_accuracy", now_batch_accuracy / self.B_S,'   ',"total_loss:",loss.item())
                # accuracy = sum(1 for pred, label in zip(predict_list, label_list) if pred == label)
                    # 计算 epoch 的平均 loss
                    # avg_loss = sum(batch_loss) / len(batch_loss)
            
            print("accuracy", accuracy / self.length,'   ',"total_loss:",sum(batch_loss))
            # print(f"Epoch {epoch} finished with Loss: {total_loss:.6f} and Accuracy: {accuracy / self.length:.4f}")

        dpo=0
        #计算公平性
        # 将 dataset 中的每个特征分离成单独的向量
        prompts_deo = [item[0] for item in self.g]  # 抽取 prompts 列
        gender_deo = [item[1] for item in self.g]   # 抽取 gender 列
        label_deo = [item[2] for item in self.g]    # 抽取 label 列
        y_pred_deo = y_pred_deo  # Predicted label (the class with the highest logit score)
        deo=DEO(s=gender_deo,y_true=label_deo,y_pred=y_pred_deo)        
        print("DEO:",deo,'\n')   
        y_pred_deo=[]             
######
######
######
###adv_model训练
        accuracy = 0
        total_loss = 0
        adv_model.train()     
        lora_model.eval()

        for param in adv_model.parameters():
            param.grad = None    
        for batch_idx, data in enumerate(self.dataset):  # 遍历数据集
            prompts = list(data[0])  # 提取 prompt 列表
            flattened_prompts = [item for prompt in prompts for item in prompt]
            label = data[2].to(self.args.device)
            label_list_str = [str(x.item()) for x in label ]#data[2] = tensor([0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0])
            
            # Tokenize inputs and prepare for batch processing
            model_inputs = self.tokenizer(flattened_prompts, return_tensors="pt", padding=True)
            model_inputs.input_ids = model_inputs.input_ids.clamp(0, lora_model.config.vocab_size - 1).to(self.args.device)#50257-1
            output_ids= self.tokenizer(label_list_str, return_tensors="pt", padding=True).to(self.args.device)
            
            model_inputs.attention_mask = model_inputs.attention_mask.to(self.args.device)
            embeddings, attention_mask = adv_model(model_inputs.input_ids, model_inputs.attention_mask)

            logits = lora_model(inputs_embeds=embeddings, attention_mask=attention_mask).logits[:, -1, :].squeeze(1)
            # 4. 根据原始 tuple 的分布，将 logits 分开
            logits_1 = logits[:len(prompts[0])]  # 前 16 个 logits 对应第一个 tuple
            logits_2 = logits[len(prompts[0]):]  # 后 16 个 logits 对应第二个 tuple
            
            loss =  -1* F.mse_loss(logits_1,logits_2)
            # 反向传播与优化
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            #adv_optimiser is SWA and there are problems with SWA, therefore we need to manually update the grads to None
            for param in adv_model.parameters():
                param.grad = None
            total_loss += loss.item()
        
        print("adv_total_loss",total_loss)
        total_loss = 0



        return lora_model.state_dict(), sum(epoch_loss) / len(epoch_loss), deo


###########
        # 训练公平性并得到公平性指标
        # for x,data in enumerate(self.gender_dataset):                
        #     for i in range(len(data[0][0])):                
        #         prompts = data[0][0][i]  # 提取第一个元素


        #         # self.gender_dataset= [([...], 'Female', 0), ([...], 'Male', 0), ([...], 'Female', 1), ...]
        #         #train mask language model
        #         model_inputs = self.tokenizer(prompts, return_tensors="pt",padding=True).to(self.args.device)
        #         # model_inputs = {k: v.to(self.args.device) for k, v in model_inputs.items()}  
        #         output_ids = model_inputs.input_ids[:,-1].to(self.args.device)
        #         model_inputs.input_ids = model_inputs.input_ids[:,:-1].to(self.args.device)
        #         model_inputs.attention_mask = model_inputs.attention_mask[:,:-1].to(self.args.device)

        #         # 得到输入的嵌入向量
        #         embeddings, attention_mask = adv_model(model_inputs.input_ids, model_inputs.attention_mask)
        #         # embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
        #         # 检查 embeddings 是否全为零
        #         # if torch.all(embeddings == 0):
        #         #     # 添加小的扰动
        #         #     noise = torch.normal(mean=torch.zeros_like(embeddings), std=1e-6 * torch.ones_like(embeddings))
        #         #     embeddings += noise
        #         # 得到输出的logits值
        #         logits = lora_model(inputs_embeds=embeddings, attention_mask=attention_mask).logits[:, -1, :].squeeze(1)
        #         loss = loss_fn(logits.view(-1, logits.size(-1)), output_ids.view(-1))
        #         loss.backward()
        #         # 梯度裁剪，防止梯度爆炸
        #         torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)    
        #         with torch.no_grad():
        #             total_loss += loss.item()
        #             print("atpresent loss:", total_loss)
        #             accuracy += torch.sum(torch.argmax(logits, dim=1) == output_ids).item() / len(output_ids)

        #         if (x + 1) % 9 == 0:
        #             optimizer.step()
        #             optimizer.zero_grad()

        # print("accuracy", accuracy / len(self.gender_dataset))
        # print("total_loss", total_loss)
###############
        # #     #二选一
        # #         # if nnn<(self.args.yita)*(self.length): 
        # #         if 1<2:
        # #             # with torch.no_grad():
        #         embeddings,attention_mask = adv_model(model_inputs.input_ids.to(self.args.device),model_inputs.attention_mask.to(self.args.device))
                                
        # #             embeddings= embeddings.to(self.args.device)
        # #             attention_mask=attention_mask.to(self.args.device)
        # #         # else:
        # #         #     # 获取 embeddings
        # #         #     with torch.no_grad():
        # #         #         # 获取 embeddings
        # #         #         embedding_layer = lora_model.get_input_embeddings().to(self.args.device)
        # #         #         embeddings = embedding_layer(model_inputs.input_ids).to(self.args.device)
        # #         #         # 获取 attention_mask
        # #         #         attention_mask = model_inputs.attention_mask.to(self.args.device)
        #         nnn= nnn+1  
        #         #第一个
                              
        #         #第二个
        #         lora_model = lora_model.to_empty(device=self.args.device).to(self.args.device)
        #         # 获取模型的 logits（自动计算 embeddings）
        #         output = lora_model(inputs_embeds=embeddings,attention_mask=attention_mask)
        #         logits = output.logits[:, -1, :].squeeze(1)  # 获取最后一个 token 的 logits
                
        #         # if torch.isnan(logits).any():
        #         #    print("logits 中包含 NaN 值")  
        #         #    logits = torch.zeros_like(logits, device=logits.device)  
        #         loss = loss_fn(logits.view(-1, logits.size(-1)), output_ids.view(-1))
        #         loss.backward()                 
        #         # **梯度裁剪**：限制梯度的范数
        #         torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
        #         # predicted_token_ids = torch.argmax(logits, dim=-1)  # 假设是预测 token IDs
        #         # Get the predicted token based on logits
        #         predicted_token_ids = get_classification(logits,self.tokenizer)
                
        #         decoded_output = self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        #         decoded_output2 = self.tokenizer.decode(output_ids.clone().detach(), skip_special_tokens=True)                 
        #         print(decoded_output,'and',decoded_output2) 
        #         # 去除空格
        #         stripped_output = decoded_output.strip()
        #         # 判断去除空格后的字符串是否是 '0' 或 '1'
        #         if stripped_output in ['0', '1']:
        #             value = int(stripped_output)  # 转换为数字
        #             y_pred_deo.append(decoded_output)
        #         else:
        #             y_pred_deo.append(999)
                                                                      
        #         # with torch.no_grad():
        #         total_loss += loss.item()
        #         accuracy += torch.sum(predicted_token_ids == output_ids).item()  
        #         if   idx == 0 or (idx+1) % 9 == 0:
        #             optimizer.step()
        #             optimizer.zero_grad()

        # torch.cuda.empty_cache()  
        # loss_avg=total_loss/ len(self.gender_dataset.dataset)
        # print("loss_avg",loss_avg,"\n")


#####adv模型训练



    def test(self, lora_model,adv_model):
        # Set model to evaluation mode
        lora_model.eval()
        test_loss = 0
        correct = 0
        total_samples = len(self.gender_dataset.dataset)


        # with torch.no_grad():
        if 1<2:
            # Test loop
            lora_model= lora_model.to(self.args.device)
            adv_model = adv_model.to(self.args.device)
            loss_fn = nn.CrossEntropyLoss()
            for idx, data in enumerate(self.gender_dataset):
                for i in range(len(data[0][0])):

                    prompts = data[0][0][i]  # 提取第一个元素
                    modified_prompts = (
                        "You are a classification model. Predict whether the annual income of the person is greater than $50k.\n"
                        "Only output a single number: `0` (for <= $50K) or `1` (for > $50K).\n"
                        + prompts
                    )
                    # modified_prompts = 'Only answer me with Number:0 or 1.\n' + prompts
                    prompts = modified_prompts
                    # self.gender_dataset= [([...], 'Female', 0), ([...], 'Male', 0), ([...], 'Female', 1), ...]
                    #train mask language model
                    model_inputs = self.tokenizer(prompts, return_tensors="pt",padding=True).to(self.args.device)
                    output_ids = model_inputs.input_ids[:,-1].to(self.args.device)
                    model_inputs.input_ids = model_inputs.input_ids[:,:-1].to(self.args.device)
                    model_inputs.attention_mask = model_inputs.attention_mask[:,:-1].to(self.args.device)
                    # 假设 model_inputs 是输入数据，确保它们都在同一个设备上
                    # model_inputs.input_ids = model_inputs.input_ids.to(self.args.device)
                    # model_inputs.attention_mask = model_inputs.attention_mask.to(self.args.device)
                    # with torch.no_grad():
                    if 1<2:
                        # embeddings,attention_mask = adv_model(model_inputs.input_ids.to(self.args.device),model_inputs.attention_mask.to(self.args.device))
                        embeddings,attention_mask = (model_inputs.input_ids.to(self.args.device),model_inputs.attention_mask.to(self.args.device))
                        
                    logits = lora_model(inputs_embeds=embeddings,attention_mask=attention_mask).logits[:,-1,:].squeeze(1)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), output_ids.view(-1))
                    test_loss += loss.item()

                    y_pred = torch.argmax(logits, dim=1)
                    correct += torch.sum(y_pred == output_ids).item()
        y=0
        # Calculate final test accuracy
        test_loss /= total_samples
        accuracy = 100. * correct / total_samples
        print(f'\nTest set: Average loss: {test_loss:.4f} \nAccuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n')

        return accuracy, test_loss

    # def save_model(self, output_dir=None, _internal_call=False):
    #     # 因为交给Trainer的model实际上是PeftModel类型，所以这里的 save_pretrained 会直接使用PeftModel的保存方法
    #     # 从而只保存 LoRA weights
    #     # 确保 LoRA 模型权重加载到设备
    #     lora_model = lora_model.to("cpu")
    #     self.model.save_pretrained(output_dir)

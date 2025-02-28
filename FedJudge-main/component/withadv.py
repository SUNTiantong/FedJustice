
import torchvision
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, LogitsProcessor
import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
from torch.amp import autocast, GradScaler
from torchcontrib.optim import SWA
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
        attn_mask=attn_mask.to(fairprompt_ids.device)#important
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


def Run_adv_model(lora_model,adv_model,adv_optimiser,dataset,tokenizer):
    
        ##adv_model训练
        accuracy = 0
        total_loss = 0


        for param in adv_model.parameters():
            param.grad = None    
        for batch_idx, data in enumerate(dataset):  # 遍历数据集
            prompts = list(data[0])  # 提取 prompt 列表 #prompts[0],prompts[1]都是单独的list。
            flattened_prompts = [item for prompt in prompts for item in prompt]
            label = data[2]
            label_list_str = [str(x.item()) for x in label ]#data[2] = tensor([0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0])

            with autocast(device_type="cuda", enabled=True): 
                # 如果损失超过1500，停止训练
                min_loss=-500
                if total_loss < min_loss:
                    print(f"Loss exceeded {min_loss}, stopping training.")
                    break 
                # Tokenize inputs and prepare for batch processing
                model_inputs = tokenizer(flattened_prompts, return_tensors="pt", padding=True)
                model_inputs.input_ids = model_inputs.input_ids.clamp(0, 50257 - 1)#50257-1
                output_ids= tokenizer(label_list_str, return_tensors="pt", padding=True)
                
                output_ids=output_ids.to("cuda:0")
                model_inputs.input_ids=model_inputs.input_ids.to("cuda:0")
                model_inputs.attention_mask = model_inputs.attention_mask.to("cuda:0")
                embeddings, attention_mask = adv_model(model_inputs.input_ids, model_inputs.attention_mask)
                with torch.no_grad():
                    logits = lora_model(inputs_embeds=embeddings, attention_mask=attention_mask).logits[:, -1, :].squeeze(1)
                # 4. 根据原始 tuple 的分布，将 logits 分开
                # logits_1 = logits[:len(prompts[0])]  # 前 16 个 logits 对应第一个 tuple
                # logits_2 = logits[len(prompts[0]):]  # 后 16 个 logits 对应第二个 tuple
                half_len = len(logits) // 2
                loss =  -1* F.mse_loss(logits[:half_len],logits[half_len:])
                # 反向传播与优化
                loss.requires_grad_(True)
                loss.backward()
            with torch.no_grad():
                if (batch_idx+1) % 5 == 0:
                # if 1:
                    adv_optimiser.step()
                    # adv_optimiser.zero_grad()
                    # print("adv_loss.item()",loss.item())
            # torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
                del logits
                torch.cuda.empty_cache()
            #adv_optimiser is SWA and there are problems with SWA, therefore we need to manually update the grads to None
            for param in adv_model.parameters():
                param.grad = None
            total_loss += loss.item()
        
        print("adv_total_loss",total_loss)
        total_loss = 0


def add_laplace_noise(embedding, noise_scale):
    """
    为输入的 embedding 添加 Laplace 噪声。

    参数：
    - embedding (torch.Tensor): 输入的 embedding 张量。
    - noise_scale (float): Laplace 噪声的强度。噪声的大小与 scale 成正比。

    返回：
    - embedding_with_noise (torch.Tensor): 添加了 Laplace 噪声后的 embedding 张量。
    """
    # 生成 Laplace 噪声，使用 torch 生成均匀分布随机数
    noise = torch.from_numpy(np.random.laplace(loc=0.0, scale=noise_scale, size=embedding.shape)).float()
    
    # 添加噪声到 embedding
    embedding_with_noise = embedding + noise
    
    return embedding_with_noise
   
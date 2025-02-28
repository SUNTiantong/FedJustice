import torch
import torch.nn as nn
from peft import (
    # prepare_model_for_int8_training,
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

class Loss(object):
    """
    所有loss的类父类
    """
    def __call__(self, model, inputs, training_args, return_outputs=False):
        """
        todo label smoothing
        用于计算loss。
        看源码发现，return_outputs=True为train时调用，return_outputs=False为eval和predict调用
        :param model: 模型
        :param inputs: 模型输入，dict
        :param training_args: 训练配置参数
        :param return_outputs:是否返回模型的输出
        :return:
        """
        raise NotImplemented


class TargetLMLoss(Loss):

    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, model, inputs, training_args, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['target_mask']
        # 模型前馈预测
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
        shift_logits = logits[..., :-1, :].contiguous()#第四个时间步时，token已经预测完毕，不用产生预测
        shift_labels = labels[..., 1:].contiguous()#第零个token已经存在，不用写入label
#logits 的形状为[batch_size, seq_len - 1, vocab_size]，eg [2, 4, 5]，表示 2 个样本，时间步数为 4，每个时间步有 5 个预测类别。那么：
# logits = [
#     [[0.1, 0.2, 0.3, 0.4, 0.5],    # 第1个样本，第1个时间步
#      [0.6, 0.7, 0.8, 0.9, 1.0],    # 第1个样本，第2个时间步
#      [1.1, 1.2, 1.3, 1.4, 1.5],    # 第1个样本，第3个时间步
#      [1.6, 1.7, 1.8, 1.9, 2.0]],   # 第1个样本，第4个时间步

#     [[2.1, 2.2, 2.3, 2.4, 2.5],    # 第2个样本，第1个时间步
#      [2.6, 2.7, 2.8, 2.9, 3.0],    # 第2个样本，第2个时间步
#      [3.1, 3.2, 3.3, 3.4, 3.5],    # 第2个样本，第3个时间步
#      [3.6, 3.7, 3.8, 3.9, 4.0]]    # 第2个样本，第4个时间步
# ]
# # 切片 logits[..., :-1, :] 会去掉每个样本的最后一个时间步，只保留前 3 个时间步，结果是：
# shift_logits = logits[..., :-1, :]  # 去掉最后一个时间步
# shift_logits = [
#     [[0.1, 0.2, 0.3, 0.4, 0.5],    # 第1个样本，第1个时间步
#      [0.6, 0.7, 0.8, 0.9, 1.0],    # 第1个样本，第2个时间步
#      [1.1, 1.2, 1.3, 1.4, 1.5]],   # 第1个样本，第3个时间步

#     [[2.1, 2.2, 2.3, 2.4, 2.5],    # 第2个样本，第1个时间步
#      [2.6, 2.7, 2.8, 2.9, 3.0],    # 第2个样本，第2个时间步
#      [3.1, 3.2, 3.3, 3.4, 3.5]]    # 第2个样本，第3个时间步
# ]
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#这里的 view(-1, shift_logits.size(-1)) 其实是 展平 shift_logits 张量的前两维（batch_size 和 seq_len - 1），并保持最后一维（vocab_size）不变。
# shift_logits.view(-1, shift_logits.size(-1)) = [
#     [0.1, 0.2, 0.3, 0.4],   # 第1个时间步
#     [0.5, 0.6, 0.7, 0.8],   # 第2个时间步
#     [0.9, 1.0, 1.1, 1.2],   # 第3个时间步
#     [1.3, 1.4, 1.5, 1.6],   # 第4个时间步
#     [1.7, 1.8, 1.9, 2.0],   # 第5个时间步
#     [2.1, 2.2, 2.3, 2.4]    # 第6个时间步
# ]

# 同样，shift_labels.view(-1) 将 shift_labels 张量展平成一维。shift_labels 的原始形状是 [2, 3]，展平后变为 [6]：
# shift_labels = [1, 2, 3, 2, 3, 0]
        return (loss, outputs) if return_outputs else loss



class TargetLMLoss_EWC(Loss):

    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, model, inputs, training_args,return_outputs=False,others=None):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['target_mask']
        # 模型前馈预测
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        All_Importance = others[0]
        Star_vals = others[1]
        i=0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad==True:
                loss += (0.5 * torch.sum(torch.mul(All_Importance[i].to(loss.device), torch.abs(parameter.data.to(loss.device) - Star_vals[i].to(loss.device))))
                        +0.5 * torch.square(torch.sum(torch.mul(All_Importance[i].to(loss.device), torch.abs(parameter.data.to(loss.device) - Star_vals[i].to(loss.device))))))
                i+=1

        return (loss, outputs) if return_outputs else loss

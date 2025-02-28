import copy
import torch
from torch import nn
import math
def process_fairness(M_deo):
    """
    处理公平性列表 M_deo,应用函数 0.8 * exp(-x)，然后归一化得到权重列表。
    
    参数:
        M_deo (list): 公平性相关的原始列表。
        
    返回:
        list: 经过处理和归一化后的权重列表。
    """
    # 1. 处理列表 M_deo
    LIST = [ math.exp(-x) for x in M_deo]
    
    # 2. 归一化 (LIST / sum(LIST))
    total = sum(LIST)
    if total != 0:
        W_list = [x / total for x in LIST]
    else:
        W_list = [1 / len(M_deo)] * len(M_deo)  # 防止除以零，总和为0时分配均匀的权重
    
    return W_list


def FairnessWeightedFedAvg(w,W_list):
    w_weighted_avg = copy.deepcopy(w[0])
    for k in w_weighted_avg.keys():
        w_weighted_avg[k] =w_weighted_avg[k] *W_list[0]
        for i in range(1, len(w)):
            w_weighted_avg[k] += w[i][k]*W_list[i]
        # w_weighted_avg[k] = torch.div(w_weighted_avg[k], len(w))
    return w_weighted_avg
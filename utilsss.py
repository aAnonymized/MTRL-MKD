import os
import pandas as pd
import numpy as np
import torch
import random


def pruning_mask(weights, mask_matrix, mask_index, k):
    _mask = mask_matrix
    tensor = weights[_mask.eq(mask_index)]
    abs_tensor = tensor.abs()
    cutoff_rank = round((1-k) * tensor.numel())
    cutoff_value = abs_tensor.cpu().kthvalue(cutoff_rank)[0].item()
    remove_mask = (weights.abs().le(cutoff_value)) * _mask.eq(mask_index)    
    # _mask[remove_mask.eq(1)] = 0
    mask = (_mask.eq(mask_index).int() - remove_mask.int())
    return mask

# 按行计算 Softmax
def row_softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

def generate_mask_matrix(arr):
    """
    生成mask矩阵，按照以下规则分配：
    - 1/4的位置为1
    - 剩下的1/3为2
    - 剩下的1/2为3
    - 其余为4
    
    参数:
        arr: 输入数组
    
    返回:
        与原数组形状相同的mask矩阵
    """
    shape = np.array(arr).shape
    total_elements = np.prod(shape)
    
    indices = np.arange(total_elements)
    np.random.shuffle(indices)
    mask = np.zeros(total_elements, dtype=int)
    
    quarter = total_elements // 3
    mask[indices[:quarter]] = 1
    
    remaining = total_elements - quarter
    third = remaining // 2
    mask[indices[quarter:quarter+third]] = 2
    
    return torch.tensor(mask.reshape(shape))



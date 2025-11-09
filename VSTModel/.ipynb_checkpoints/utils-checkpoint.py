import os
import pandas as pd
import numpy as np
import torch
import random
import pickle
from torch import nn
from config import PatchMerging_module_idx, WindowAttention3D_module_idx, Mlp_module_idx

def get_alpha(epoch, max_epoch, start=0.9, end=0.1):
    """
    alpha 调度函数 (线性下降)
    epoch: 当前的 epoch
    max_epoch: 总 epoch 数
    start: 初始 alpha
    end:   最终 alpha
    """
    alpha = start + (start - end) * (epoch / max_epoch)
    alpha = np.min([alpha, 1.0])
    return max(end, alpha)  # 确保不小于 end

def pruning_mask(weights, mask_matrix, mask_index, k):
    _mask = mask_matrix
    tensor = weights[_mask.eq(mask_index)]
    abs_tensor = tensor.abs()
    cutoff_rank = round(k * tensor.numel())
    cutoff_value = abs_tensor.cpu().kthvalue(cutoff_rank)[0].item()
    remove_mask = (weights.abs().le(cutoff_value)) * _mask.eq(mask_index)    
    # _mask[remove_mask.eq(1)] = 0
    mask = (_mask.eq(mask_index).int() - remove_mask.int())
    return mask

def new_pruning_mask(weights, student_weights, mask_matrix, mask_index, k):
    _mask = mask_matrix
    # dif_weights = (weights - student_weights)
    dif_weights = weights
    tensor = dif_weights[_mask.eq(mask_index)]
    abs_tensor = tensor.abs()
    cutoff_rank = round(k * tensor.numel())
    cutoff_value = abs_tensor.cpu().kthvalue(cutoff_rank)[0].item()
    remove_mask = (dif_weights.abs().le(cutoff_value)) * _mask.eq(mask_index) 
    # remove_mask = (weights.abs().le(cutoff_value)) * _mask.eq(mask_index)    
    mask = (_mask.eq(mask_index).int() - remove_mask.int())
    return mask, dif_weights

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
    fusion_num = total_elements // 2
    total_elements -= fusion_num
    
    quarter = int(total_elements * 0.5)
    mask[indices[:quarter]] = 1
    
    remaining = total_elements - quarter
    third = remaining
    mask[indices[quarter:quarter+third]] = 2

    return torch.tensor(mask.reshape(shape))


def load_weights(models, phase, fpath):
    for i in range(len(models)):
        if phase == 'train':
            print(f'train phase: load weightd from local file pth.')
            _pretrained_dict = torch.load(fpath)
            pretrained_dict = {}
            for k, v in _pretrained_dict.items():
                if k.startswith('module.'):
                    new_key = k.replace('module.', '')
                elif k.startswith('cls_head.'):
                    new_key = k.replace('cls_head.', '')
                else:
                    new_key = k 
                pretrained_dict[new_key] = v
        else:
            if phase == 'test':
                print(f'test phase: load weightd from local file pth.')
                _pretrained_dict = torch.load(fpath)
                pretrained_dict = {}
                for k, v in _pretrained_dict.items():
                    if k.startswith('module.'):
                        new_key = k.replace('module.', '')
                    # 移除 'cls_head.' 前缀
                    elif k.startswith('cls_head.'):
                        new_key = k.replace('cls_head.', '')
                    else:
                        new_key = k  # 如果没有前缀，则保持不变

                    pretrained_dict[new_key] = v
        model_dict = models[i].state_dict()
        # 过滤掉不匹配的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        missing, unexpected = models[i].load_state_dict(pretrained_dict, strict=False)
        print(f'No.{i+1} model load pretrained weighted end. Missing keys:", {len(missing)}')
        
def PI(models, phase):
    if phase == 'test':
        print(f'load mask_matrix from local file.')
        with open(f'./mask_matrix.npz', 'rb') as f:
            mask_matrix_dict = pickle.load(f)
            return mask_matrix_dict

    if phase == 'train':
        print(f'start to generate mask_matrix.')
        mask_matrix_dict = {}
        for module_idx, module in enumerate(models[0].modules()):
            if module_idx in PatchMerging_module_idx or module_idx in Mlp_module_idx: continue
            if hasattr(module, "qkv"):
                if isinstance(module.qkv, nn.Linear):
                    if module.qkv.weight.data.numel() < 50000: continue
                    mask = torch.ByteTensor(module.qkv.weight.data.size()).fill_(0)
                    mask_matrix_dict[module_idx] = generate_mask_matrix(mask.numpy())
            else:
                if isinstance(module, nn.Linear):
                    if module.weight.data.numel() < 50000: continue
                    mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                    mask_matrix_dict[module_idx] = generate_mask_matrix(mask.numpy())
        print(f'save mask_matrix to local file.')
        with open(f'./mask_matrix.npz', 'wb') as f:
            pickle.dump(mask_matrix_dict, f)
        return mask_matrix_dict
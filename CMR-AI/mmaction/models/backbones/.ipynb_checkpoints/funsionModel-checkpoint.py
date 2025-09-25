import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.layers import DropPath, trunc_normal_
import sys

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
import matplotlib.pyplot as plt

class fusion_ConcatHead(nn.Module):

    def __init__(self,
                 num_classes=9,
                 in_channels=1024,
                 num_mod=2,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__()

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_mod = num_mod
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(in_channels*num_mod, num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # print(x.shape)
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            for i in range(self.num_mod):
                x[i] = self.avg_pool(x[i])
        # [N, in_channels, 1, 1, 1]
        x = torch.cat(x, dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
    
from abc import ABCMeta, abstractmethod
class funsionModel(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 model_list=None,
                 num_class=9
                ):
        super().__init__()
        self.model_list = model_list
        self.funsion_head = fusion_ConcatHead(num_classes=num_class, num_mod=len(model_list))

    def contrast_loss(self, x1, x2):
        squared_diff = (x1-x2)**2
        sum_squared_diff = torch.sum(squared_diff)
        contrast_loss = torch.sqrt(sum_squared_diff)
        return contrast_loss
    
    def forward(self, imgs):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x = []
        for i in range(len(self.model_list)):
            imgs1 = imgs[i]
            model = self.model_list[i]
            x1 = model(imgs1)
            x.append(x1)
        return self.funsion_head(x)

def load_from_pretrained(model_list, model_list_weights):
    def _load(path):
        _pretrained_dict = torch.load(path)
        pretrained_dict = {}
        for k, v in _pretrained_dict['state_dict'].items():
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
            # 移除 'cls_head.' 前缀
            elif k.startswith('cls_head.'):
                new_key = k.replace('cls_head.', '')
            else:
                new_key = k  # 如果没有前缀，则保持不变
        
            pretrained_dict[new_key] = v
        return pretrained_dict
    for index, model in enumerate(model_list):
        model_dict = model.state_dict()
        pretrained_dict = _load(model_list_weights[index])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        # print(pretrained_dict)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f'Success load preTrain weights . ')
        # for p in model.parameters():  ## 把参数冻住，fine-tune时不进行梯度更新.
        #     p.requires_grad = False
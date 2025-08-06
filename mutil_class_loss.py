import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    
    def __init__(self, device, gamma = 1.0):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = torch.tensor(gamma, dtype = torch.float32).to(device)
        self.eps = 1e-6        
    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').to(self.device)
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()*10
     
    
from sklearn.metrics import roc_auc_score
def cal_auc(label, predict):
    class_roc_auc_list = []
    useful_classes_roc_auc_list = []
    for i in range(label.shape[1]):
        class_roc_auc = roc_auc_score(label[:, i], predict[:, i])
        class_roc_auc_list.append(class_roc_auc)
        # if i != 0:
        useful_classes_roc_auc_list.append(class_roc_auc)
    return np.mean(np.array(useful_classes_roc_auc_list))
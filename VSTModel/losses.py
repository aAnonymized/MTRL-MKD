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

def kd_loss(student_logits, teacher_probs, y_true, T=2.0, alphas=[1, 1], eps=1e-5, one_hot=True):
    """
    student_logits: [batch, num_classes] 学生 logits
    teacher_probs:  [batch, num_classes] 教师概率分布 (可能比较尖锐)
    y_true:         [batch] 类别索引，或者 [batch, num_classes] one-hot
    """
    # 处理标签
    if one_hot:
        y_true = y_true.argmax(dim=1)   # 转成 index
    
    # 硬标签的交叉熵
    ce = F.cross_entropy(student_logits, y_true)

    # 学生分布 (log_softmax)
    log_p_s_T = F.log_softmax(student_logits / T, dim=1)

    # 教师分布 + 防止零
    p_t = (teacher_probs + eps) / (teacher_probs + eps).sum(dim=1, keepdim=True)
    # 升温（让分布更平滑）
    p_t_T = torch.pow(p_t, 1.0 / T)
    p_t_T = p_t_T / p_t_T.sum(dim=1, keepdim=True)

    # KL 散度 (teacher || student)
    kld = F.kl_div(log_p_s_T, p_t_T, reduction='batchmean') * (T * T)
    return alphas[0]*ce + alphas[1] * kld

def agent_kd_loss(student_logits, teacher_probs, y_true, T=2.0, alphas=[1, 1], eps=1e-5, one_hot=True):
    """
    student_logits: [batch, num_classes] 学生 logits
    teacher_probs:  [batch, num_classes] 教师概率分布 (可能比较尖锐)
    y_true:         [batch] 类别索引，或者 [batch, num_classes] one-hot
    """
    # 处理标签
    if one_hot:
        y_true = y_true.argmax(dim=1)   # 转成 index
    
    # 硬标签的交叉熵
    ce = F.cross_entropy(student_logits, y_true)

    # 学生分布 (log_softmax)
    log_p_s_T = F.log_softmax(student_logits / T, dim=1)

    # 教师分布 + 防止零
    p_t = (teacher_probs + eps) / (teacher_probs + eps).sum(dim=1, keepdim=True)
    # 升温（让分布更平滑）
    p_t_T = torch.pow(p_t, 1.0 / T)
    p_t_T = p_t_T / p_t_T.sum(dim=1, keepdim=True)

    # KL 散度 (teacher || student)
    kld = (F.kl_div(log_p_s_T, p_t_T, reduction='none') * (T*T)).sum(dim=1)
    return alphas[0] * ce + alphas[1] * kld
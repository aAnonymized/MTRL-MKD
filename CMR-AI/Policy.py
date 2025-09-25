import torch
import torch.nn as nn
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class Policy(nn.Module):
    def __init__(self, input_size, teacher_num):
        super(Policy, self).__init__()
        self.teacher_num = teacher_num
        self.steam = nn.Sequential(
                nn.Linear(input_size, 128, bias=False),
                nn.ReLU())
        
        self.logit_head = nn.Linear(128, teacher_num, bias=True)                
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, student_info):
        out = self.steam(student_info)
        logit_weights = self.softmax(self.logit_head(out))
        # logit_weights = self.sigmoid(self.logit_head(out))
        return logit_weights

def train_agent(epoch, agent, student_infos, agent_rewards, logits_agent_actions, agent_optimizer):
    agent.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')
    for student_info, rewards, action in zip(student_infos, agent_rewards, logits_agent_actions):
        ### Π(a|s, r) ---> max(r)  表示的是在s状态下动作a，获得的奖励.
        agent_pred = agent(student_info)
        agent_optimizer.zero_grad() 
        action_label = torch.ones_like(agent_pred).detach().cpu()
        loss_logits = F.binary_cross_entropy(agent_pred, action_label, weight=rewards.unsqueeze(-1))
        loss = loss_logits + 0
        loss.backward()
        agent_optimizer.step()
        
        agent_loss.update(loss.item(), action.size(0))
        
    print('Epoch:{}, agent Loss:{:.6f}'.format(epoch, agent_loss.avg))

import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from abc import ABCMeta, abstractmethod
import torch.nn.functional as F
import os
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
from skimage import transform
import numpy as np
from torch import nn
import SimpleITK as sitk
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from VSTModel.swinTransformer3D import SwinTransformer3D
from VSTModel.losses import FocalLoss, kd_loss, agent_kd_loss
from VSTModel.weighted_auc_f1 import get_weighted_auc_f1, cal_auc
from VSTModel.load_dataset import ACDC
from VSTModel.utils import pruning_mask, row_softmax, new_pruning_mask, get_alpha, load_weights, PI
from VSTModel.Policy import Policy, train_agent
from VSTModel.get_data import get_data

TRAIN = 'train'
TEST  = 'test'
phase = 'train'

def main(args):
    ## Task Split
    num_class = {
            0: ['HCM', 'DCM', 'RV', 'MINF'],
            1: ['HCM', 'DCM'], 
            2: ['RV', 'MINF'],
        }

    save_models = {0: 'full', 1: '1', 2: '2'}
    dummy_labels = num_class[0]
    models = []
    for _, v in num_class.items():
        num_class_ = len(v)
        models.append(SwinTransformer3D(num_class=num_class_))
    print(f'Total model is {len(models)}')

    ## load model weights.
    load_weights(models, TRAIN, args.pretrain_path)
    ## PI
    mask_matrix_dict = PI(models, TRAIN)
    print(mask_matrix_dict.keys())

    ## RL agent
    input_size, teacher_num = 0, len(models)-1
    for name, param in list(models[0].named_parameters())[-2:-1]:
        print(f"layer anem: {name} | size: {param.size()}")
        input_size = param.size()[-1]
    agent = Policy(input_size=input_size, teacher_num=teacher_num)

    if phase == TRAIN:
        mask_list = []
        for module_idx, modules in enumerate(zip(models[0].modules(), models[1].modules(), models[2].modules())):
            if module_idx not in mask_matrix_dict.keys():  continue
            for mask_index, module in enumerate(modules):
                if hasattr(module, "qkv"):
                    if mask_index == 0: 
                        pass
                    else:
                        with torch.no_grad():
                            module.qkv.weight[mask_matrix_dict[module_idx] != mask_index] = \
                                module.qkv.weight[mask_matrix_dict[module_idx] != mask_index].detach().requires_grad_(False)
                else:
                    if mask_index == 0: 
                        pass
                    else:
                        with torch.no_grad():
                            module.weight[mask_matrix_dict[module_idx] != mask_index] = \
                                module.weight[mask_matrix_dict[module_idx] != mask_index].detach().requires_grad_(False)
    train_data, test_data, dataset_list = get_data(args.train_data_csv, 
                                                   args.test_data_csv, 
                                                   dummy_labels)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        for i in range(len(models)):
            models[i] = nn.DataParallel(models[i])
        
    for i in range(len(models)):
        models[i] = models[i].cuda()

    fn_loss  = FocalLoss(device = 'cuda:0', gamma = 2.).to('cuda:0')
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    cross_loss = nn.CrossEntropyLoss()

    optimizers = []
    for i in range(len(models)):
        optimizers.append(torch.optim.SGD(models[i].parameters(), lr=args.base_lr))
    agent_optimizer = torch.optim.SGD(agent.parameters(), lr=args.base_lr)

    train_acdc_data = ACDC(data=train_data, phase = 'train', img_size=(224, 224))
    train_data_loader = DataLoader(train_acdc_data, batch_size=args.batch_size, shuffle=True, num_workers=6)
    test_acdc_data = ACDC(data=test_data, phase = 'test', img_size=(224, 224))
    test_data_loader = DataLoader(test_acdc_data, batch_size=args.batch_size, shuffle=True, num_workers=6)

    if phase == TRAIN:
        writer = SummaryWriter(log_dir='./runs/VSTMTRL-ACDC')
    
    suffix = 0
    if phase == TRAIN:
        import time
        pre_mutil_teacher_acc = []
        now_mutil_teacher_acc = []
        for i in range(1, len(models)):
            pre_mutil_teacher_acc.append(0)
            now_mutil_teacher_acc.append(0)
        logit_actions_list = []
        agent_rewars_list  = []
        student_infos_list = []    
        mask_list_dict= {}
        bast_acc = 0.0
        for epoch_num in range(0, args.max_epoch):
            alpha = get_alpha(epoch_num, args.max_epoch)
            print(f"--------> epoch_num: {epoch_num}")
            train_loader_nums = len(train_data_loader.dataset)
            probs = np.zeros((train_loader_nums, len(num_class[0])), dtype = np.float32)
            gt    = np.zeros((train_loader_nums, len(num_class[0])), dtype = np.float32)
            k_index = 0
            start_time = time.time()
            total_train_loss = 0.0
            correct = 0.0
            for i in range(len(models)): models[i].train()
            logit_action = []
            mutil_teacher_correct = []
            mutil_teacher_num = []
            for i in range(1, len(models)):
                mutil_teacher_correct.append(0)
                mutil_teacher_num.append(0)
            for batch_idx, (batch_data, batch_finding, batch_label) in enumerate(train_data_loader):
                if batch_idx == 0 and epoch_num == 0: rl_weights = [1, 1]
                weighted_list = []
                mutil_teacher_label = torch.zeros_like(batch_label)
                student_predicts = torch.zeros_like(batch_label).cuda()
                student_output = torch.zeros_like(batch_label)
                pre_lables = 0
                s_train_loss = torch.tensor(0.0, device='cuda')
                t_Attn_states_outs = []
                for i, k in enumerate(num_class.keys()):
                    if i == 0: 
                        continue
                    else:
                        ## train teacher.
                        teacher_train_data_index = pd.Series(batch_finding).isin(num_class[k])
                        teacher_train_data_index = teacher_train_data_index.to_numpy()
                        weighted_list.append(np.sum(teacher_train_data_index > 0))
                        mutil_teacher_num[i-1] += weighted_list[i-1]
                        t_train_data = batch_data[teacher_train_data_index]
                        t_train_label = batch_label[teacher_train_data_index][:, pre_lables:pre_lables+len(num_class[k])]
                        if np.sum(teacher_train_data_index > 0) == 0: 
                            pre_lables += len(num_class[i])
                            continue
                        t_output, _, t_Attn_states_out = models[i](t_train_data.cuda())
                        t_Attn_states_out = [arr.detach() for arr in t_Attn_states_out] 
                        t_Attn_states_outs.append(t_Attn_states_out)
                        mutil_teacher_label[teacher_train_data_index, pre_lables:pre_lables+len(num_class[k])] = \
                                                row_softmax(t_output.cpu().detach())
                        t_output = t_output.reshape(t_output.shape[0], -1)
                        t_train_label = t_train_label.reshape(t_train_label.shape[0], -1).cuda()
                        t_train_loss = fn_loss(t_output, t_train_label)
                        optimizers[i].zero_grad()
                        t_train_loss.backward()
                        optimizers[i].step()
                        predicted_ = torch.argmax(t_output, 1)
                        labels_ = torch.argmax(t_train_label.cuda(), 1)
                        correct_ = (predicted_ == labels_).sum().item() 
                        mutil_teacher_correct[i-1] += correct_
                        pre_lables += len(num_class[k])
                        
                ## RL-guied train student
                s_output, student_info, _ = models[0](batch_data.cuda())
                s_output = s_output.reshape(s_output.shape[0], -1)
                log_s_output = torch.nn.LogSoftmax(dim=1)(student_output)
                s_train_loss = kd_loss(s_output.cuda(), mutil_teacher_label[teacher_train_data_index].cuda(), batch_label[teacher_train_data_index].cuda(), args.T, alphas=[0.963, 0.037])     
                optimizers[0].zero_grad()
                s_train_loss.backward()
                optimizers[0].step()
                total_train_loss += s_train_loss
                
                predicted = torch.argmax(s_output, 1)
                labels = torch.argmax(batch_label.cuda(), 1)
                correct += (predicted == labels).sum().item()  
                
                student_infos_list.append(student_info.detach().cpu())
                with torch.no_grad():
                    logit_action = agent(student_info.detach().cpu())
                if epoch_num == 0:
                    logit_action = torch.ones_like(logit_action).cuda()
                logit_actions_list.append(logit_action.detach().cpu())                
                agent_reward = - agent_kd_loss(s_output.cuda(), mutil_teacher_label.cuda(), batch_label.cuda(), T, alphas=rl_weights)
                rewards_mean = agent_reward.mean() 
                rewards_std = agent_reward.std() 
                normalized_reward = (agent_reward - rewards_mean) / rewards_std 
                normalized_reward = normalized_reward.detach().cpu()
                normalized_reward = torch.clamp(normalized_reward, min=0, max=1)
                agent_rewars_list.append(normalized_reward)
                mean_logit_action = torch.mean(logit_action, dim=0, keepdim=True)  # shape=(1, 4)
                rl_weights = mean_logit_action[0]
                mean_logit_action.cpu().numpy()

                ## MD
                if epoch_num != 0 and (batch_idx+1) % (len(train_data_loader.dataset)//args.batch_size//3) == 0:
                    for module_index, modules in enumerate(zip(models[0].modules(), models[1].modules(), models[2].modules())):
                        mask_list = []
                        student_share_mask_list = []
                        teacher_class_num_sum = 0
                        if module_index not in mask_matrix_dict.keys():  continue
                        if isinstance(modules[0], nn.Linear):
                            if modules[0].weight.data.numel() < 50000: continue
                            for mask_index, module in enumerate(modules):
                                if mask_index == 0: 
                                    for i in range(1, len(models)):
                                        if hasattr(modules[0], "qkv"):
                                            weights = module.qkv.weight.data
                                            student_weights = modules[i].qkv.weight.data
                                        else:
                                            weights = module.weight.data
                                            student_weights = modules[i].weight.data
                                        mask, _ = new_pruning_mask(weights.cpu(), student_weights.cpu(), mask_matrix_dict[module_index], mask_index, k=1)
                                else:
                                    if hasattr(modules[0], "qkv"):
                                        weights = module.qkv.weight.data
                                        student_weights = modules[0].qkv.weight.data
                                    else:
                                        weights = module.weight.data
                                        student_weights = modules[0].weight.data
                                mask,  diff_weight = new_pruning_mask(weights.cpu(), student_weights.cpu(), mask_matrix_dict[module_index], mask_index, k=1)
                                mask_list.append(mask.cuda())
                            all_weights_mask = torch.ones_like(modules[0].weight.data)
                            for i in range(1, len(models)):
                                unique_weights_mask_ = mask_list[i-1]
                                unique_weights_mask = (unique_weights_mask_ >= 1).int()
                                ## new add.
                                if hasattr(modules[0], "qkv"):
                                    ### teacher -> student
                                    modules[0].qkv.weight.grad.data[unique_weights_mask].fill_(0)
                                    unique_weights = unique_weights_mask * modules[i].qkv.weight.data 
                                    modules[0].qkv.weight.data = (modules[0].qkv.weight.data)*(all_weights_mask-unique_weights_mask) + \
                                                            + args.momentum*(unique_weights_mask * modules[0].qkv.weight.data) + (1-args.momentum)*unique_weights
                                else:
                                ### teacher -> student
                                    modules[0].weight.grad.data[unique_weights_mask].fill_(0)
                                    unique_weights = unique_weights_mask * modules[i].weight.data 
                                    modules[0].weight.data = (modules[0].weight.data)*(all_weights_mask-unique_weights_mask) + \
                                                            + args.momentum*(unique_weights_mask * modules[0].weight.data) + (1-args.momentum)*unique_weights
                            mask_list.clear()

                if batch_idx != 0 and batch_idx % (len(train_data_loader)-1) == 0:  
                    print(f'student_infos_list:{student_infos_list[0].shape}, agent_rewars_list:{agent_rewars_list[0].shape}, logit_actions_list:{logit_actions_list[0].shape}')
                    train_agent(epoch=epoch_num, agent=agent, student_infos=student_infos_list, agent_rewards=agent_rewars_list, logits_agent_actions=logit_actions_list, agent_optimizer=agent_optimizer)
                    all_logits = torch.stack(logit_actions_list, dim=0)  # (num_steps, batch, actions)
                    col_mean = all_logits.mean(dim=(0,1))  # (2,)
                    print(f'logit_actions_list : {col_mean}')
                    logit_actions_list.clear()
                    agent_rewars_list.clear()
                    student_infos_list.clear()
                    
            logit_actions_list.clear()
            agent_rewars_list.clear()
            student_infos_list.clear()
            
            for i in range(len(now_mutil_teacher_acc)):
                now_mutil_teacher_acc[i] = mutil_teacher_correct[i]/mutil_teacher_num[i]
        
            print(f'mutil-teacher acc is : {now_mutil_teacher_acc}')
            print(f"epoch_num {epoch_num} train loss {total_train_loss} ")  

            writer.add_scalars('Training Metrics', {
                'Loss': total_train_loss,
                'Accuracy': correct / train_loader_nums,
            }, epoch_num)
            
            lr_ = args.base_lr*(1-args.decay_rate)
            T = T*(1-args.decay_rate)
            for i in range(len(optimizers)):
                for param_group in optimizers[i].param_groups:
                    param_group['lr'] = lr_
            
            mutil_teacher_compare_result = (np.array(now_mutil_teacher_acc) > np.array(pre_mutil_teacher_acc))
            for i in range(mutil_teacher_compare_result.shape[0]):
                if mutil_teacher_compare_result[i]: 
                    pre_mutil_teacher_acc[i] = now_mutil_teacher_acc[i]
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            test_interval = 1
            if (epoch_num + 1) % test_interval == 0:
                test_loader_nums = len(test_data_loader.dataset)
                test_probs = np.zeros((test_loader_nums, len(num_class[0])), dtype = np.float32)
                test_gt    = np.zeros((test_loader_nums, len(num_class[0])), dtype = np.float32)
                test_k  = 0
                models[0].eval()
                with torch.no_grad():
                    for test_data_batch, _, test_label_batch in test_data_loader:
                        test_data_batch = test_data_batch.cuda()
                        test_label_batch = test_label_batch.cuda()
                        test_outputs, _ = models[0](test_data_batch)
                        test_outputs = test_outputs.reshape(test_outputs.shape[0], -1)           
                        test_label_batch = test_label_batch.reshape(test_outputs.shape[0], -1)
                        # storing model predictions for metric evaluation 
                        test_probs[test_k: test_k + test_outputs.shape[0], :] = test_outputs.cpu().detach().numpy()
                        test_gt[   test_k: test_k + test_outputs.shape[0], :] = test_label_batch.cpu().detach().numpy()
                        test_k += test_outputs.shape[0]
                    test_label = np.argmax(test_gt, axis=1)
                    test_pred = np.argmax(test_probs, axis=1)
                    print(f"auc: {cal_auc(test_gt, test_probs)} | acc: {np.sum(test_label==test_pred)/test_k}")
                    if (np.sum(test_label==test_pred)/test_k) >= bast_acc:
                        bast_acc = (np.sum(test_label==test_pred)/test_k)
                        suffix += 1
                        if suffix >= 20:
                            for i in range(len(models)):
                                os.makedirs(f'./train_model/MTRL-MKD-SKD/', exist_ok=True)
                                os.makedirs(f'./train_model/MTRL-MKD-SKD/{save_models[i]}', exist_ok=True)
                                save_mode_path = os.path.join(f'./train_model/MTRL-MKD-SKD/{save_models[i]}', f'best_model_{suffix}.pth')
                                torch.save(models[i].state_dict(), save_mode_path)
                                print(f"save model {save_mode_path} ...")  

if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a MTRL-MD for action localization')
    parser.add_argument('pretrain_path', type=str, metavar='PATH',
                        help='path to a pretrain_path')
    parser.add_argument('train_data_csv', type=str, metavar='PATH',
                        help='path to train data csv path')
    parser.add_argument('test_data_csv', type=str, metavar='PATH',
                        help='path to test data csv path')
    parser.add_argument('base_lr', default=0.01, type=float,
                        help='init learning rate')
    parser.add_argument('decay_rate', default=0.0009, type=float,
                        help='decay_rate')
    parser.add_argument('batch_size', default=8, type=int,
                        help='batch size')
    parser.add_argument('max_epoch', default=400, type=int,
                        help='max epoch number')
    parser.add_argument('momentum', default=0.99, type=float,
                        help='momentum value')
    parser.add_argument('T', default=2.0, type=float,
                        help='tempt')
    args = parser.parse_args()
    
    main(args)

import sys
import torch
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

from VSTModel.swinTransformer3D import SwinTransformer3D
from VSTModel.weighted_auc_f1 import get_weighted_auc_f1
from VSTModel.load_dataset import ACDC
from VSTModel.get_data import get_data

phase = 'test'
if __name__ == '__main__':
    ## Task Split
    num_class = {
            0: ['HCM', 'DCM', 'RV', 'MINF'],
            1: ['HCM', 'DCM'], 
            2: ['RV', 'MINF'],
        }
    dummy_labels = num_class[0]
    model = SwinTransformer3D(num_class=len(dummy_labels))  

    ## load model.
    _pretrained_dict = torch.load('')
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
    model_dict = model.state_dict()
    # 过滤掉不匹配的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
    print(f'model load pretrained weighted end. Missing keys:", {len(missing)}')
    train_data, test_data, dataset_list = get_data('', '', dummy_labels)
    test_acdc_data = ACDC(data=test_data, phase = 'test', img_size=(224, 224))
    test_data_loader = DataLoader(test_acdc_data, batch_size=8, shuffle=True, num_workers=6)

    total_acc_list = []
    total_auroc_list = []
    total_weight_auroc_list = []
    total_weight_acc_list = []
    ### eval.
    for i in range(10):
        test_loader_nums = len(test_data_loader.dataset)
        test_probs = np.zeros((test_loader_nums, len(dummy_labels)), dtype = np.float32)
        test_gt    = np.zeros((test_loader_nums, len(dummy_labels)), dtype = np.float32)
        test_k  =0
        model.eval()
        with torch.no_grad():
            for test_data_batch, _, test_label_batch in test_data_loader:
                test_data_batch = test_data_batch.cuda()
                test_label_batch = test_label_batch.cuda()
                test_outputs, _, _ = model(test_data_batch.cuda())
                test_outputs = test_outputs.reshape(test_outputs.shape[0], -1)           
                test_label_batch = test_label_batch.reshape(test_outputs.shape[0], -1)
                test_probs[test_k: test_k + test_outputs.shape[0], :] = test_outputs.cpu().detach().numpy()
                test_gt[   test_k: test_k + test_outputs.shape[0], :] = test_label_batch.cpu().detach().numpy()
                test_k += test_outputs.shape[0]
            test_label = np.argmax(test_gt, axis=1)
            test_pred = np.argmax(test_probs, axis=1)
            weight_auc, auc_list = get_weighted_auc_f1(test_probs, test_pred, test_label)

            cm = confusion_matrix(test_label, test_pred)
            acc_list = []
            weighted_acc = 0.0
            for i in range(len(dataset_list)):
                weight = dataset_list[i] / sum(dataset_list)
                correct = cm[i][i]
                acc = float(correct) / dataset_list[i]
                acc_list.append(acc)
                weighted_acc += weight*acc 
            
            total_auroc_list.append(auc_list)
            total_acc_list.append(acc_list)
            total_weight_auroc_list.append(weight_auc)
            total_weight_acc_list.append(weighted_acc)
    print(total_weight_auroc_list)
    print(total_weight_acc_list)

    auc_arr = np.array(total_auroc_list)
    print(auc_arr.shape)
    for i in range(auc_arr.shape[-1]):
        auc_arr_cls = auc_arr[:, i]
        mean = np.mean(auc_arr_cls)
        std = np.std(auc_arr_cls)
        print(mean, std)

    acc_arr = np.array(total_acc_list)
    print(acc_arr.shape)
    for i in range(auc_arr.shape[-1]):
        acc_arr_cls = acc_arr[:, i]
        mean = np.mean(acc_arr_cls)
        std = np.std(acc_arr_cls)
        print(mean, std)

    np.mean(total_weight_acc_list), np.std(total_weight_acc_list)
    np.mean(total_weight_auroc_list), np.std(total_weight_auroc_list)
    
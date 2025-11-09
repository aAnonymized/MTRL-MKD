# calculate auc, f1-score, specificity, sensitivity and their confidence intervals by bootstrap method
# for multilabel tasks
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def cal_auc(label, predict):
    class_roc_auc_list = []
    useful_classes_roc_auc_list = []
    for i in range(label.shape[1]):
        class_roc_auc = roc_auc_score(label[:, i], predict[:, i])
        class_roc_auc_list.append(class_roc_auc)
        # if i != 0:
        useful_classes_roc_auc_list.append(class_roc_auc)
    return np.mean(np.array(useful_classes_roc_auc_list))

def AUC(data):
    """
    Calculate AUC（for bootstrap）
    :param data: data[:, 0] = score, data[:, 1] = label, data[:, 2] = index
    :return: auc score
    """
    pos_index = int(data[0, 2])
    fpr, tpr, thersholds = roc_curve(data[:, 1].astype(int).tolist(), data[:, 0].tolist(), pos_label=pos_index)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def f1_scores(data):
    """
    Calculate f1 score（for bootstrap）
    :param data: data[:, 0] = pred, data[:, 1] = label, data[:, 2] = index
    :return: f1 score
    """
    label_f1 = data[:, 1]
    pred_f1 = data[:, 0]
    conf_mat = confusion_matrix(y_true=label_f1, y_pred=pred_f1)
    pos_index = int(data[0, 2])
    tp = conf_mat[pos_index][pos_index]
    fp = np.sum(conf_mat[:, pos_index]) - tp
    fn = np.sum(conf_mat[pos_index]) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_cls = 2 * precision * recall / (precision + recall)
    return f1_cls

def f1_scores_all(data):
    label_f1 = data[:, 1]
    pred_f1 = data[:, 0]
    f1_cls = f1_score(label_f1, pred_f1, average='weighted')
    return f1_cls

def get_specificity(data):
    """
    Calculate specificity（for bootstrap）
    :param data: data[:, 0] = score, data[:, 1] = label, data[:, 2] = index
    :return: specificity (when sensitivity=0.9)
    """
    pos_index = int(data[0, 2])
    fpr, tpr, thersholds = roc_curve(data[:, 1].astype(int).tolist(), data[:, 0].tolist(), pos_label=pos_index)
    fpr = fpr.tolist()
    tpr = tpr.tolist()
    for i in range(len(tpr)):
        if tpr[i] == 0.9:
            return 1 - fpr[i]
        elif tpr[i] < 0.9 and tpr[i + 1] > 0.9:
            a = 0.9 - tpr[i]
            b = tpr[i + 1] - 0.9
            fpr_i = a / (a + b) * fpr[i + 1] + b / (a + b) * fpr[i]
            return 1 - fpr_i


def get_sensitivity(data):
    """
    Calculate sensitivity（for bootstrap）
    :param data: data[:, 0] = score, data[:, 1] = label, data[:, 2] = index
    :return: sensitivity (when specificity=0.9)
    """
    pos_index = int(data[0, 2])
    fpr, tpr, thersholds = roc_curve(data[:, 1].astype(int).tolist(), data[:, 0].tolist(), pos_label=pos_index)
    fpr = fpr.tolist()
    tpr = tpr.tolist()
    for i in range(len(fpr)):
        if fpr[i] == 0.1:
            return tpr[i]
        elif fpr[i] < 0.1 and fpr[i + 1] > 0.1:
            a = 0.1 - fpr[i]
            b = fpr[i + 1] - 0.1
            tpr_i = a / (a + b) * tpr[i + 1] + b / (a + b) * tpr[i]
            return tpr_i


def bootstrap(data, B, c, func):
    """
    Calculate bootstrap confidence interval
    :param data: array, sample data
    :param B: Sampling times normally, B>=1000
    :param c: Confidence level, for example, 0.95
    :param func: estimator
    :return: upper and lower bounds of bootstrap confidence interval
    """
    array = np.array(data)
    n = len(array)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)
        data_sample = array[index_arr]
        sample_result = func(data_sample)
        sample_result_arr.append(sample_result)

    a = 1 - c
    k1 = int(B * a / 2)
    k2 = int(B * (1 - a / 2))
    auc_sample_arr_sorted = sorted(sample_result_arr)
    lower = auc_sample_arr_sorted[k1]
    higher = auc_sample_arr_sorted[k2]

    return lower, higher


def get_weighted_auc_f1(test_probs, test_pred, test_label):
    weighted_auroc = 0
    weighted_F1 = 0
    auc_list = []
    f1_list  = []
    ### test_label, test_pred, test_probs
    for index in range(test_probs.shape[-1]):
        data = np.zeros((test_probs.shape[0], 3))
        score = test_probs[:, index]
        data[:, 0] = score
        data[:, 1] = test_label
        data[:, 2] = index

        data_f1 = np.zeros((test_probs.shape[0], 3))
        data_f1[:, 0] = test_pred
        data_f1[:, 1] = test_label
        data_f1[:, 2] = index

        # Input must be a data
        roc_auc = AUC(data)
        auc_list.append(roc_auc)
        f1_score_ = f1_scores(data_f1)
#         sensitivity = get_sensitivity(data)
#         specificity = get_specificity(data)

#         # bootstrap for 1000, calculate 95% CI
#         bootstrap_num = 1000
#         # bootstrap_num = 500
#         result = bootstrap(data, bootstrap_num, 0.95, AUC)
#         result2 = bootstrap(data_f1, bootstrap_num, 0.95, f1_scores)
#         result3 = bootstrap(data, bootstrap_num, 0.95, get_sensitivity)
#         result4 = bootstrap(data, bootstrap_num, 0.95, get_specificity)

        weight = np.sum(test_label == index) / test_probs.shape[0]
        
        weighted_auroc += weight*roc_auc
        weighted_F1 += weight*f1_score_
    print('--------------------------------------------------')
    print(f'auc_list : {auc_list}')
    print('weighted_auroc: ', weighted_auroc)
    print('weighted_F1: ', weighted_F1)
    return weighted_auroc, auc_list

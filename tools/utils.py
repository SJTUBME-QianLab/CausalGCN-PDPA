import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy import interp
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
import seaborn as sns
import itertools
from itertools import cycle
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')  # 设置字体为Times New Roman
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(args, flag='simple'):
    if isinstance(args, str):
        args_new = eval(args)
        if flag == 'simple':
            return args_new
        elif flag == 'deep':
            return str2list(args_new, flag=flag)
    elif isinstance(args, dict):
        if flag == 'simple':
            return args
        elif flag == 'deep':
            for i, v in args.items():
                if isinstance(v, str):
                    args[i] = eval(v)
            return args
    else:
        raise ValueError('wrong arg type')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_graph_name(node_type=None, edge_type=None, dist_type='gau', sparse=None, adj_norm='DAD'):
    if sparse is not None:
        assert isinstance(sparse, dict)
        sparse_name = f"{sparse['level']}" + \
                      (f"th{sparse['thresh']}" if sparse['rate'] is None else f"ra{sparse['rate']}")
        name = f"N{node_type}_E{edge_type}_D{dist_type}_S{sparse_name}_{adj_norm}"
    else:
        name = f"N{node_type}_E{edge_type}_D{dist_type}_S{sparse}_{adj_norm}"
    # if edge_type is None:
    #     name = f"N{node_type}_D{dist_type}_{adj_norm}"
    # else:
    #     name = f"N{node_type}_E{edge_type}_D{dist_type}_{adj_norm}"
    return name


def get_emb_name(feature_depth, kernels, epoch, seed, fold):
    if epoch is None and seed is None and fold is None:
        name = 'C{}_K{}'.format(
            '.'.join([str(s) for s in feature_depth]),
            '.'.join([str(s) for s in kernels])
        )
    else:
        name = 'C{}_K{}_e{:d}_split{:d}_f{:d}'.format(
            '.'.join([str(s) for s in feature_depth]),
            '.'.join([str(s) for s in kernels]),
            epoch, seed, fold
        )
    return name


def onehot_code(Y, num_class):
    Y = np.array(Y)
    Yc_onehot = np.zeros((len(Y), num_class))
    for i in range(num_class):
        Yc_onehot[np.where(Y == i)[0], i] = 1.0
    return Yc_onehot


def eval_metric_cl2(true, prob):
    num_class = int(max(true) + 1)
    assert num_class in prob.shape
    assert num_class == 2
    pred = list(prob.values.argmax(axis=0 if num_class == prob.shape[0] else 1))
    assert len(true) == len(pred)

    # acc
    acc = sum([pred[i] == true[i] for i in range(len(true))]) / len(true)

    # confusion matrix
    con_matrix = np.array(
        [[sum([pred[i] == k1 and true[i] == k2 for i in range(len(true))]) for k1 in range(num_class)] for k2 in range(num_class)])
    tn, fp, fn, tp = con_matrix.ravel()
    con_arr = con_matrix.ravel()

    SEN_cal = lambda tn, fp, fn, tp: tp / (tp + fn) if (tp + fn) != 0 else 0
    PRE_cal = lambda tn, fp, fn, tp: tp / (tp + fp) if (tp + fp) != 0 else 0
    SPE_cal = lambda tn, fp, fn, tp: tn / (tn + fp) if (tn + fp) != 0 else 0
    NPV_cal = lambda tn, fp, fn, tp: tn / (tn + fn) if (tn + fn) != 0 else 0
    #     Fbeta_cal = lambda pre, sen, beta: (1+beta*beta)*pre*sen / (beta*beta*pre+sen)

    sen = SEN_cal(*con_arr)
    pre = PRE_cal(*con_arr)
    spe = SPE_cal(*con_arr)
    npv = NPV_cal(*con_arr)
    f1 = 2 * pre * sen / (pre + sen)
    #     beta = 1
    #     fbeta = Fbeta_cal(pre, sen, beta)

    evals = {
        'confusion_matrix': con_matrix,
        'acc': acc, 'accuracy': acc,
        'pre': pre, 'precision': pre, 'ppv': pre,
        'npv': npv,
        'sen': sen, 'sensitivity': sen, 'recall': sen, 'tpr': sen,
        'spe': spe, 'specificity': spe, 'tnr': spe,
        'fpr': 1-spe,
        'f1': f1, 'f1_score': f1, 'f1score': f1,
    }

    return evals


def get_auc_cl2(true, prob):
    num_class = int(max(true) + 1)
    assert num_class == 2
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    # y_label = onehot_code(true, num_class)
    prob = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True).repeat(prob.shape[-1], axis=1)

    fpr, tpr, _ = roc_curve(true, prob[:, 1])
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def eval_metric(true, prob):
    num_class = int(max(true) + 1)
    assert num_class in prob.shape
    pred = list(prob.values.argmax(axis=0 if num_class == prob.shape[0] else 1))
    assert len(true) == len(pred)

    # acc
    acc = sum([pred[i] == true[i] for i in range(len(true))]) / len(true)

    # confusion matrix
    con_matrix = np.array(
        [[sum([pred[i] == k1 and true[i] == k2 for i in range(len(true))]) for k1 in range(num_class)] for k2 in range(num_class)])

    #     con_arr1 = np.zeros((num_class, 4))
    #     for k in range(num_class):
    #         tp = con_matrix[k, k]
    #         fp = np.sum(con_matrix[:, k]) - tp
    #         fn = np.sum(con_matrix[k, :]) - tp
    #         tn = np.sum(con_matrix) - tp - fp - fn
    #         print(tn, fp, fn, tp)
    #         con_arr1[k, :] = [tn, fp, fn, tp]

    con_arr = np.zeros((num_class, 4))
    for k in range(num_class):
        tp = sum([pred[i] == k and true[i] == k for i in range(len(true))])
        fp = sum([pred[i] == k and true[i] != k for i in range(len(true))])
        tn = sum([pred[i] != k and true[i] != k for i in range(len(true))])
        fn = sum([pred[i] != k and true[i] == k for i in range(len(true))])
        # print(tn, fp, fn, tp)
        con_arr[k, :] = [tn, fp, fn, tp]

    SEN_cal = lambda tn, fp, fn, tp: tp / (tp + fn) if (tp + fn) != 0 else 0
    PRE_cal = lambda tn, fp, fn, tp: tp / (tp + fp) if (tp + fp) != 0 else 0
    SPE_cal = lambda tn, fp, fn, tp: tn / (tn + fp) if (tn + fp) != 0 else 0
    NPV_cal = lambda tn, fp, fn, tp: tn / (tn + fn) if (tn + fn) != 0 else 0
    #     Fbeta_cal = lambda pre, sen, beta: (1+beta*beta)*pre*sen / (beta*beta*pre+sen)

    # macro 分别求出每个类，再进行算术平均
    sen = np.nansum(np.array([SEN_cal(*cc) for cc in con_arr])) / num_class
    pre = np.nansum(np.array([PRE_cal(*cc) for cc in con_arr])) / num_class
    spe = np.nansum(np.array([SPE_cal(*cc) for cc in con_arr])) / num_class
    npv = np.nansum(np.array([NPV_cal(*cc) for cc in con_arr])) / num_class
    f1 = 2 * pre * sen / (pre + sen)
    #     beta = 1
    #     fbeta = Fbeta_cal(pre, sen, beta)

    # micro 全局计算。将所有混淆矩阵累加在一起，然后计算
    sen_mi = SEN_cal(*list(np.sum(con_arr, axis=0)))
    pre_mi = PRE_cal(*list(np.sum(con_arr, axis=0)))
    spe_mi = SPE_cal(*list(np.sum(con_arr, axis=0)))
    npv_mi = NPV_cal(*list(np.sum(con_arr, axis=0)))
    f1_mi = 2 * pre_mi * sen_mi / (pre_mi + sen_mi)

    evals = {
        'macro': {
            'confusion_matrix': con_matrix,
            'acc': acc, 'accuracy': acc,
            'pre': pre, 'precision': pre, 'ppv': pre,
            'npv': npv,
            'sen': sen, 'sensitivity': sen, 'recall': sen, 'tpr': sen,
            'spe': spe, 'specificity': spe, 'tnr': spe,
            'fpr': 1-spe,
            'f1': f1, 'f1_score': f1, 'f1score': f1,
        },
        'micro': {
            'confusion_matrix': con_matrix,
            'acc': acc, 'accuracy': acc,
            'pre': pre_mi, 'precision': pre_mi, 'ppv': pre_mi,
            'npv': npv_mi,
            'sen': sen_mi, 'sensitivity': sen_mi, 'recall': sen_mi, 'tpr': sen_mi,
            'spe': spe_mi, 'specificity': spe_mi, 'tnr': spe_mi,
            'fpr': 1-spe_mi,
            'f1': f1_mi, 'f1_score': f1_mi, 'f1score': f1_mi,
        }
    }

    # sen1 = recall_score(true, pred, average='macro', zero_division=0)
    # pre1 = precision_score(true, pred, average='macro', zero_division=0)
    # f1score1 = f1_score(true, pred, average='macro', zero_division=0)
    # sen1_mi = recall_score(true, pred, average='micro', zero_division=0)
    # pre1_mi = precision_score(true, pred, average='micro', zero_division=0)
    # f1score1_mi = f1_score(true, pred, average='micro', zero_division=0)
    # print('myself macro', pre, sen, f1)
    # print('sklearn macro', pre1, sen1, f1score1)
    # print('myself micro', pre_mi, sen_mi, f1_mi)
    # print('sklearn micro', pre1_mi, sen1_mi, f1score1_mi)

    return evals


def get_auc(true, prob):
    num_class = int(max(true) + 1)
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    y_label = onehot_code(true, num_class)
    prob = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True).repeat(prob.shape[-1], axis=1)

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc, fpr, tpr


def get_CM(true, prob):
    num_class = int(max(true) + 1)
    assert num_class in prob.shape
    pred = list(prob.values.argmax(axis=0 if num_class == prob.shape[0] else 1))
    assert len(true) == len(pred)

    # con_matrix = np.array([[sum([pred[i] == k1 and true[i] == k2 for i in range(len(true))]) for k1 in range(num_class)] for k2 in range(num_class)])
    # acc = sum([pred[i] == true[i] for i in range(len(true))]) / len(true)
    con_matrix = sklearn.metrics.confusion_matrix(true, pred)
    acc = sklearn.metrics.accuracy_score(true, pred)
    print(con_matrix)
    print(acc)

    return con_matrix


def plot_confusion_matrix(name, out_path, cm, classes, cmap=plt.cm.GnBu):  # PuBu
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    num_cm = cm
    f, ax = plt.subplots()
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # 设置坐标轴
    ax.set_ylabel('True Label', family="Times New Roman", weight="bold", size=20)
    ax.set_xlabel('Predicted Label', family="Times New Roman", weight="bold", size=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # 设置colorbar
    cb = f.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=24)
    labels = cb.ax.get_xticklabels() + cb.ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes, rotation=90)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部

    # add number
    ax.set_ylim(len(classes) - 0.5, -0.5)
    # fmt = '.2f' if normalize else 'd'
    thresh = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(num_cm[i, j], 'd'),
                horizontalalignment="center", verticalalignment="bottom",
                color="white" if cm[i, j] > thresh else "black",
                family="Times New Roman", fontsize=24)
        ax.text(j, i, '({:.2f})'.format(cm[i, j]),
                horizontalalignment="center", verticalalignment="top",
                color="white" if cm[i, j] > thresh else "black",
                family="Times New Roman", fontsize=18)

    ax.spines['left'].set_linewidth(1.5)  # 设置左部坐标轴的粗细
    ax.spines['right'].set_linewidth(1.5)  # 设置右边坐标轴的粗细
    ax.spines['bottom'].set_linewidth(1.5)  # 设置底部坐标轴的粗细
    ax.spines['top'].set_linewidth(1.5)  # 设置上部坐标轴的粗细

    ax.set_title(name, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    # plt.show()


# if __name__ == '__main__':
#     main()

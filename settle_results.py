# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 1:03
# @Author  : tangxl
# @FileName: settle_results.py
# @Software: PyCharm
"""
"""

import numpy as np
import pandas as pd
import os
import re
import pickle
import platform
import nibabel as nib
from tools.utils import eval_metric_cl2, get_auc_cl2, eval_metric, get_auc, get_CM, plot_confusion_matrix
import sklearn
import seaborn as sns
import itertools
from itertools import cycle
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')  # 设置字体为Times New Roman
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

raw_dirs = {
    'windows': {
        'template_path': 'D://soft//mri_tools//spm12//toolbox//AAL3//AAL3v1_1mm.nii.gz',
        'related_nii_path': 'D://OneDrive - sjtu.edu.cn//PD_data//PD_exp//remote_data//QSM//aal3v1_roi//roi_index_all.nii.gz',
        'data_path': 'D://OneDrive - sjtu.edu.cn//PD_data//PD_exp//remote_data//QSM_voxel//',
        'save_path': 'D://OneDrive - sjtu.edu.cn//PD_data//PD_exp//local_results//1001_CNNpre_causal_ratio'
    },
    'linux': {
        'template_path': '/home/data/tangxl/software-pack/spm12/toolbox/AAL3/AAL3v1_1mm.nii.gz',
        'related_nii_path': '/home/data/tangxl/PD_exp/QSM/aal3v1_roi/roi_index_all.nii.gz',
        'data_path': '/home/data/tangxl/PD_exp/QSM_voxel/',
        # 'save_path': '/home/data/tangxl/PD_exp/QSM_results/1001_CNNpre_causal_ratio',
        'save_path': '/home/data/tangxl/PD_exp/QSM_results/1009_Ours_all',
    },
}
raw_dirs = raw_dirs[platform.system().lower()]
metrics = ['acc', 'pre', 'sen', 'spe', 'f1', 'auc']
ave = ''


def main():
    main_param()


def main_param():  # 0820
    # concat 5fold
    data_name = 'posture.120.1_2_0n0_pw11_r0.3'
    # data_name = 'posture.120.1_2_0n0_pw9_r0.4'
    # data_name = 'posture.120.1_2_0n0_pw7_r0.5'

    split_list = [int(kk.split('split')[1]) for kk in os.listdir(os.path.join(raw_dirs['save_path'], data_name))]
    # split_list = [2018, 2020, 2022, 2025, 2027, 2028, 2031, 2032, 2039, 2049]
    # split_list = [2027, 2031, 2032, 2039]
    # split_list = list(range(2020, 2023))
    # split_list = list(range(2038, 2040))
    # split_list = [2020]

    concat_split_seed(split_list, data_name)

    # seed_list = [2022]
    # concat_fold(split_list, seed_list, data_name)
    # concat_all(list(range(2020, 2040)), data_name)
    concat_all(split_list, data_name)

    # rename(split_list, data_name)


def rename(split_list, data_name):
    dir_i = os.path.join(raw_dirs['save_path'], 'concat', data_name)
    os.makedirs(dir_i, exist_ok=True)
    for split in split_list:
        seed_list = [int(kk.split('seed')[1]) for kk in
                     os.listdir(os.path.join(raw_dirs['save_path'], data_name, f"split{split}"))]
        for seed in seed_list:
            for fold in range(5):
                save_path = os.path.join(raw_dirs['save_path'], data_name, f'split{split}', f'seed{seed}', f'fold{fold}')
                if not os.path.exists(save_path):
                    continue
                for para_name in sorted(os.listdir(save_path)):
                    # if 'La0.000L10.000L20.000' in para_name:
                    #     os.system(f'rm -r {os.path.join(save_path, para_name)}')
                    dir_ii = os.path.join(save_path, para_name)
                    new_name = dir_ii.replace('L1', '0L1').replace('L2', '0L2').replace('__raw', '0__raw')
                    # print(dir_ii)
                    os.rename(dir_ii, new_name)


def concat_split_seed(split_list, data_name):
    anchor = {2020: [0.72115, 0.17541, 0.19446],
              2027: [0.72115, 0.14079, 0.14997],
              2031: [0.72596, 0.13653, 0.13758],
              2032: [0.72115, 0.10503, 0.11001],
              2039: [0.72596, 0.10806, 0.11555]}
    dir_i = os.path.join(raw_dirs['save_path'], 'concat', data_name)
    os.makedirs(dir_i, exist_ok=True)
    out_dir = os.path.join(dir_i, 'CM_5CV')
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(raw_dirs['data_path'], data_name.split('_pw')[0], data_name)
    for split in split_list:
        seed_list = [int(kk.split('seed')[1]) for kk in
                     os.listdir(os.path.join(raw_dirs['save_path'], data_name, f"split{split}"))]
        if os.path.isfile(os.path.join(dir_i, f'finalscores_5CV_para_split{split}.pkl')):
            with open(os.path.join(dir_i, f'finalscores_5CV_para_split{split}.pkl'), 'rb') as f:
                evals_dict = pickle.load(f)
        else:
            evals_dict = dict()
        for seed in seed_list:
            save_path = os.path.join(raw_dirs['save_path'], data_name, f'split{split}', f'seed{seed}')
            if not os.path.exists(os.path.join(save_path, 'fold0')):
                continue
            for para_name in sorted(os.listdir(os.path.join(save_path, 'fold0'))):
                para_name0 = '__'.join(para_name.split('__')[1:])
                if f'{para_name0}_SEED{seed}' in evals_dict.keys():
                    continue

                scores_5CV = pd.DataFrame()
                flag = 0
                for fold in range(5):
                    if not os.path.exists(os.path.join(save_path, f'fold{fold}')):
                        break
                    ff = [kk for kk in os.listdir(os.path.join(save_path, f'fold{fold}')) if para_name0 in kk]
                    if len(ff) != 1:
                        print(f'{save_path}, fold{fold}, {para_name0} not exist')
                        break
                    dir_ii = os.path.join(save_path, f'fold{fold}', ff[0])
                    ss = SettleResults(data_dir, dir_ii)
                    score_i = ss.get_final_score()
                    if score_i is None:
                        print('Not complete 5 fold')
                        break
                    scores_5CV = pd.concat([scores_5CV, score_i], axis=1)
                    flag += 1
                if flag < 5:
                    continue

                data_dir0 = '/home/data/tangxl/PD_exp/QSM/Y_posture.120.1_2_0n0_pw_5/Y_posture.120.1_2_0n0_pw_5_pnum_200'
                with open(os.path.join(data_dir0, 'label.pkl'), 'rb') as f:
                    label_all, sub_name_all = pickle.load(f)
                label_df = pd.DataFrame({'label': label_all, 'sub_name': sub_name_all})
                Fassign_5CV = catch_CausalScore(para_name=para_name0, save_path=raw_dirs['save_path'],
                                                split=split, seed=seed, mode='test')
                causal = label_df.merge(Fassign_5CV, on='sub_name', how='left')
                causal0_std = np.std(causal[causal['label'] == 0].iloc[:, 2:].values, axis=0).mean()  # 负样本方差
                causal1_std = np.std(causal[causal['label'] == 1].iloc[:, 2:].values, axis=0).mean()  # 正样本方差

                auc, _, _ = get_auc_cl2(true=scores_5CV.iloc[-1, :], prob=scores_5CV.iloc[:-1, :])
                evals = eval_metric_cl2(true=scores_5CV.iloc[-1, :], prob=scores_5CV.iloc[:-1, :])
                cm = get_CM(scores_5CV.iloc[-1, :], scores_5CV.iloc[:-1, :])
                print(split, seed, para_name0)
                classes = ['0', 'n0'] if data_name.split('_')[2] == '0n0' else [str(kk) for kk in
                                                                                data_name.split('_')[2]]
                plot_confusion_matrix(data_name + '\n' + para_name0,
                                      os.path.join(out_dir, f'{para_name0}_split{split}_seed{seed}.png'),
                                      cm=cm, classes=classes)

                evals_dict.update({f'{para_name0}_SEED{seed}': [evals[kk] for kk in metrics[:-1]] + [auc, causal0_std, causal1_std]})
                # if evals['acc'] >= anchor[split][0] and causal0_std <= anchor[split][1] and causal1_std <= anchor[split][2]:
                #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', split, seed)

        with open(os.path.join(dir_i, f'finalscores_5CV_para_split{split}.pkl'), 'wb') as f:
            pickle.dump(evals_dict, f)
        evals_df = []
        for ii, vv in evals_dict.items():
            evals_df.append([split, int(ii.split('SEED')[1]), ii.split('_SEED')[0]] + list(vv))
        evals_df = pd.DataFrame(evals_df, columns=['split', 'seed', 'para_name'] +
                                                  [kk if kk[:3] == 'acc' else f'{kk}' for kk in metrics] + ['ca0', 'ca1'])
        evals_df.to_csv(os.path.join(dir_i, f'finalscores_5CV_para_split{split}.csv'), index=False)


def concat_all(split_list, data_name):
    dir_i = os.path.join(raw_dirs['save_path'], 'concat', data_name)
    metrics_seeds = pd.DataFrame()
    for split in split_list:
        if not os.path.isfile(os.path.join(dir_i, f'finalscores_5CV_para_split{split}.csv')):
            continue
        df = pd.read_csv(os.path.join(dir_i, f'finalscores_5CV_para_split{split}.csv'))
        metrics_seeds = pd.concat([metrics_seeds, df], axis=0)
    metrics_seeds.to_csv(os.path.join(dir_i, 'search.csv'), index=False)

    # anchor = {2020: [0.72115, 0.17541, 0.19446],
    #           2027: [0.72115, 0.14079, 0.14997],
    #           2031: [0.72596, 0.13653, 0.13758],
    #           2032: [0.72115, 0.10503, 0.11001],
    #           2039: [0.72596, 0.10806, 0.11555]}
    # find = []
    # for i in range(len(metrics_seeds)):
    #     split, seed, acc, causal0_std, causal1_std = metrics_seeds.iloc[i, [0, 1, 3, 9, 10]]
    #     if acc >= anchor[split][0] and causal0_std <= anchor[split][1] and causal1_std <= anchor[split][2]:
    #         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', split, seed)
    #         find.append(metrics_seeds.iloc[i, :])
    # find = pd.DataFrame(find, columns=metrics_seeds.columns)
    # find.to_csv(os.path.join(dir_i, 'find_CausalScore.csv'), index=False)


def catch_CausalScore(para_name, save_path, split, seed, mode='test'):
    # split, seed = 2018, 921
    # para_name = 'G16.16.32.32.32_r0.90__all_ycloseinter_l2_dot_La0.0100L10.0500L20.1000__raw'
    # save_path = r'D:\OneDrive - sjtu.edu.cn\PD_data\PD_exp\remote_data\QSM_results\1005_Ours_all'
    save_pathi = os.path.join(save_path, 'posture.120.1_2_0n0_pw11_r0.3', f'split{split}', f'seed{seed}')
    sub_names, Fassign_5CV = [], []
    for fold in range(5):
        files = [kk for kk in os.listdir(os.path.join(save_pathi, f'fold{fold}')) if para_name in kk]
        if len(files) != 1:
            print(os.path.join(save_pathi, f'fold{fold}'))
        assert len(files) == 1
        pkl_path = os.path.join(save_pathi, f'fold{fold}', files[0], 'epoch', f'final_{mode}_Fassign.pkl')
        with open(pkl_path, 'rb') as f:
            Fassign = pickle.load(f)
        for sub, vv in Fassign.items():
            Fassign_5CV.append((vv - vv.min()) / (vv.max() - vv.min()))
#             Fassign_5CV.append(vv)
            sub_names.append(sub)
    #     Fassign_5CV.extend([[kk] + list(vv.T[0]) for kk, vv in Fassign.items()])
    # Fassign_5CV = pd.DataFrame(Fassign_5CV)
    Fassign_5CV = pd.concat([pd.DataFrame({'sub_name': sub_names}), pd.DataFrame(np.hstack(Fassign_5CV)).T], axis=1)
    tt = Fassign_5CV.groupby('sub_name').mean()
    tt.insert(loc=0, column='sub_name', value=tt.index)
    tt.reset_index(drop=True, inplace=True)
    return tt


def concat_seed(split_list, data_name, para_name):
    dir_i = os.path.join(raw_dirs['save_path'], 'concat', data_name)
    columns = ['split', 'seed'] + [kk if kk[:3] == 'acc' else f'{kk}' for kk in metrics]
    metrics_seeds = pd.DataFrame()
    for split in split_list:
        if not os.path.isfile(os.path.join(dir_i, f'finalscores_5CV_para_split{split}.csv')):
            continue
        df = pd.read_csv(os.path.join(dir_i, f'finalscores_5CV_para_split{split}.csv'))
        df = df[df['para_name'] == para_name]
        if len(df) == 0:
            continue
        metrics_seeds = pd.concat([metrics_seeds, df], axis=0)
    metrics_seeds.to_csv(os.path.join(dir_i, para_name+'.csv'), index=True)


def concat_fold(split_list, seed_list, data_name):
    dir_i = os.path.join(raw_dirs['save_path'], 'concat', data_name)
    os.makedirs(dir_i, exist_ok=True)
    for split in split_list:
        # seed_list = [int(kk.split('seed')[1]) for kk in
        #              os.listdir(os.path.join(raw_dirs['save_path'], data_name, f"split{split}"))]
        for seed in seed_list:
            # if os.path.isfile(os.path.join(dir_i, f'finalscores_5CV_para_split{split}_seed{seed}.pkl')):
            # with open(os.path.join(dir_i, f'finalscores_5CV_para_split{split}_seed{seed}.pkl'), 'rb') as f:
            #     scores = pickle.load(f)
            # else:
            scores = [{}] * 5
            for fold in range(5):
                save_path = os.path.join(raw_dirs['save_path'], data_name, f'split{split}', f'seed{seed}', f'fold{fold}')
                if not os.path.exists(save_path):
                    break
                for para_name in sorted(os.listdir(save_path)):
                    para_name0 = '__'.join(para_name.split('__')[1:])
                    # para_name0 = 'Nemb' + para_name.split('_Nemb')[1]
                    # if para_name0 == 'debug':
                    #     continue
                    dir_ii = os.path.join(save_path, para_name)
                    data_dir = os.path.join(raw_dirs['data_path'], data_name.split('_pw')[0], data_name)
                    ss = SettleResults(data_dir, dir_ii, para_name)
                    score_i = ss.get_final_score()
                    if score_i is None:
                        continue
                    tmp_dict = scores[fold].copy()
                    tmp_dict.update({para_name0: score_i})
                    scores[fold] = tmp_dict

            out_dir = os.path.join(dir_i, 'CM_5CV')
            os.makedirs(out_dir, exist_ok=True)
            evals_dict = {}
            for para_name0 in scores[0].keys():
                scores_5CV = pd.DataFrame()
                if sum([para_name0 in scores[kk].keys() for kk in range(5)]) < 5:
                    print(para_name0, 'missing fold')
                    continue
                for fold in range(5):
                    vv = scores[fold][para_name0]
                    scores_5CV = pd.concat([scores_5CV, vv], axis=1)
                auc, _, _ = get_auc_cl2(true=scores_5CV.iloc[-1, :], prob=scores_5CV.iloc[:-1, :])
                evals = eval_metric_cl2(true=scores_5CV.iloc[-1, :], prob=scores_5CV.iloc[:-1, :])
                evals_dict.update({para_name0: [evals[kk] for kk in metrics[:-1]] + [auc]})
                cm = get_CM(scores_5CV.iloc[-1, :], scores_5CV.iloc[:-1, :])
                print(split, seed, para_name0)
                classes = ['0', 'n0'] if data_name.split('_')[2] == '0n0' else [str(kk) for kk in data_name.split('_')[2]]
                plot_confusion_matrix(data_name + '\n' + para_name0,
                                      os.path.join(out_dir, f'{para_name0}_split{split}_seed{seed}.png'),
                                      cm=cm, classes=classes)
            evals_df = pd.DataFrame(evals_dict, index=[kk if kk[:3] == 'acc' else f'{kk}' for kk in metrics])
            evals_df.to_csv(os.path.join(dir_i, f'finalscores_5CV_para_split{split}_seed{seed}.csv'), index=True)


def check():
    seed = 1
    # for data_name in os.listdir(raw_dirs['save_path']):
    #     if data_name in ['concat']:
    #         continue
    for data_name in ['Y_posture.120.1_2_0n0_pw_5_r0.4']:
        for fold in range(5):
            save_path = os.path.join(raw_dirs['save_path'], data_name, f'seed{seed}', f'fold{fold}')
            if not os.path.exists(save_path):
                continue
            for para_name in sorted(os.listdir(save_path)):
                dir_ii = os.path.join(save_path, para_name)
                if os.path.isfile(os.path.join(dir_ii, 'CM.png')):
                    continue
                # print('not exist: \t', dir_ii)
                data_dir = os.path.join(raw_dirs['data_path'], data_name.split('_pw')[0], data_name)
                ss = SettleResults(data_dir, dir_ii, para_name)
                # ss.concat_epoch_scores(epoch=100, att=False)
                if ss.concat_trend_scores(num_epoch=30, metrics=metrics, phase='test', ave=ave):
                    # os.system('rm -R %s' % os.path.join(save_path, para_name))
                    print('not complete ...... removed %s' % dir_ii)
                    continue
                ss.merge_pkl(num_epoch=30, type_list=['test_score', 'train_eval_score'])
                ss.confusion_matrix(out_path=os.path.join(dir_ii, 'CM.png'))


class SettleResults:
    def __init__(self, data_dir, exp_dir, para_name=''):
        self.atlas_nii = nib.load(raw_dirs['template_path'])
        self.atlas_arr = self.atlas_nii.get_fdata()
        self.mask_affine = self.atlas_nii.affine.copy()
        self.mask_header = self.atlas_nii.header.copy()
        self.mask_header.set_qform(self.mask_affine)
        self.mask_header.set_sform(self.mask_affine)
        self.related_roi = nib.load(raw_dirs['related_nii_path'])
        self.related_roi_arr = self.related_roi.get_fdata()
        self.data_dir = data_dir
        self.exp_dir = exp_dir
        self.data_name = os.path.split(self.data_dir)[-1]
        self.para_name = '__'.join(para_name.split('__')[1:])

        self.num_class = int(self.data_name.split('_')[1])
        self.patch_size = int(re.search('pw\d+', self.data_name).group().split('pw')[1])
        self.rate = float(self.data_name.split('_r')[1])

        self.coor = np.load(os.path.join(self.data_dir, 'coordinates.npy'))

        with open(os.path.join(self.data_dir, 'label.pkl'), 'rb') as f:
            label_all, sub_name_all = pickle.load(f)
        label_dict = dict(zip(sub_name_all, label_all))
        self.label_df = pd.DataFrame(label_dict, index=['true_label'])

    def concat_epoch_scores(self, epoch, att=False, phase='test'):
        if len(phase) == 1:
            with open(os.path.join(self.exp_dir, "epoch", "epoch%d_%s_score.pkl" % (epoch, phase)), 'rb') as f:
                score_df = pd.DataFrame(pickle.load(f))
        elif len(phase) == 2:
            with open(os.path.join(self.exp_dir, "epoch", "epoch%d_%s_score.pkl" % (epoch, phase)), 'rb') as f:
                score_df = pickle.load(f)
            with open(os.path.join(self.exp_dir, "epoch", "epoch%d_test_score.pkl" % epoch), 'rb') as f:
                score_df = score_df.update(pickle.load(f))
            score_df = pd.DataFrame(score_df)
        else:
            raise ValueError
        score_df.index = ['score'+str(kk) for kk in range(len(score_df))]
        if att:
            with open(os.path.join(self.exp_dir, "epoch", "epoch%d_train_eval_Apatch.pkl" % epoch), 'rb') as f:
                att_tr = pickle.load(f)
            with open(os.path.join(self.exp_dir, "epoch", "epoch%d_test_Apatch.pkl" % epoch), 'rb') as f:
                att_te = pickle.load(f)
            att_tr_df = pd.DataFrame(att_tr)
            att_te_df = pd.DataFrame(att_te)
            att_df = pd.concat([att_tr_df, att_te_df], axis=1)
            scores = pd.concat([score_df, self.label_df, att_df], axis=0)
            scores.to_csv(os.path.join(self.exp_dir, "all_sample_final_score_att.csv"), index=True)
        else:
            scores = pd.concat([score_df, self.label_df], axis=0)
            scores.to_csv(os.path.join(self.exp_dir, "all_sample_final_score.csv"), index=True)

    def concat_trend_scores(self, num_epoch, metrics, start_epoch=0, phase='test', ave=''):
        with open(os.path.join(self.data_dir, 'label.pkl'), 'rb') as f:
            label_all, sub_name_all = pickle.load(f)
        label_dict = dict(zip(sub_name_all, label_all))
        label_df = pd.DataFrame(label_dict, index=['true_label'])
        if os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch1_test_score.pkl')):
            if not os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch%d_%s_score.pkl' % (num_epoch, phase))):
                print('------------------%s does not complete!' % self.exp_dir)
                return True
            evals_list = []
            for epo in range(1+start_epoch, 1+num_epoch):
                with open(os.path.join(self.exp_dir, 'epoch', 'epoch%d_%s_score.pkl' % (epo, phase)), 'rb') as f:
                    score_pred = pickle.load(f)
                # with open(os.path.join(self.exp_dir, 'epoch', 'epoch%d_%s_score.pkl' % (epo, phase)), 'rb') as f:
                #     score_pred.update(pickle.load(f))
                score_pred_df = pd.DataFrame(score_pred)
                scores = pd.concat([score_pred_df, label_df], axis=0).dropna(axis=1)
                if self.num_class == 2:
                    evals = eval_metric_cl2(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
                    evals_list.append([evals[kk] for kk in metrics])
                else:
                    evals = eval_metric(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
                    evals_list.append([evals[ave][kk] for kk in metrics])
            evals_df = pd.DataFrame(evals_list, columns=[kk if kk[:3] == 'acc' else f'{kk}_{ave}' for kk in metrics])
            evals_df.to_csv(os.path.join(self.exp_dir, "trend_metrics_%s.csv" % phase), index=False)
        elif os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'allepo_%d_test_score.pkl' % num_epoch)):
            with open(os.path.join(self.exp_dir, 'epoch', 'allepo_%d_%s_score.pkl' % (num_epoch, phase)), 'rb') as f:
                all_score_pred = pickle.load(f)
            evals_list = []
            for epo, score_pred in all_score_pred.items():
                score_pred_df = pd.DataFrame(score_pred)
                scores = pd.concat([score_pred_df, label_df], axis=0).dropna(axis=1)
                if self.num_class == 2:
                    evals = eval_metric_cl2(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
                    evals_list.append([evals[kk] for kk in metrics])
                else:
                    evals = eval_metric(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
                    evals_list.append([evals[ave][kk] for kk in metrics])
            evals_df = pd.DataFrame(evals_list, columns=[kk if kk[:3] == 'acc' else f'{kk}_{ave}' for kk in metrics])
            evals_df.to_csv(os.path.join(self.exp_dir, "trend_metrics_%s.csv" % phase), index=False)
        elif not os.path.isfile(os.path.join(self.exp_dir, 'log.txt')):
            print('------------------%s missing log.txt!' % self.exp_dir)
            return True  # failed

    def merge_pkl(self, num_epoch, type_list, start_epoch=0):
        # type_list = ['svm', 'test_score', 'train_eval_score']
        for tt in type_list:
            if os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'allepo_%d_%s.pkl' % (num_epoch - start_epoch, tt))):
                print('------------------%s have merged!' % tt)
                continue
            elif not os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch1_%s.pkl' % tt)):
                if not os.path.isfile(os.path.join(self.exp_dir, 'log.txt')):
                    print('------------------%s missing log.txt!' % tt)
                    return True  # failed
            if not os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch%d_%s.pkl' % (num_epoch, tt))):
                print('------------------%s does not complete!' % tt)
                return False
            tt_all = {}
            for epo in range(1+start_epoch, 1+num_epoch):
                with open(os.path.join(self.exp_dir, 'epoch', 'epoch%d_%s.pkl' % (epo, tt)), 'rb') as f:
                    score_pred = pickle.load(f)
                tt_all[epo] = score_pred
            with open(os.path.join(self.exp_dir, 'epoch', 'allepo_%d_%s.pkl' % (len(tt_all), tt)), 'wb') as f:
                pickle.dump(tt_all, f)
        # print('save %s %s' % (self.exp_dir, tt))
        assert set([kk for kk in os.listdir(os.path.join(self.exp_dir, 'epoch')) if kk[:3] == 'all']) \
               == set(['allepo_%d_%s.pkl' % (num_epoch - start_epoch, tt) for tt in type_list])
        if platform.system().lower() == 'linux' and os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch1_test_score.pkl')):
            os.system('rm %s/epoch*.pkl' % os.path.join(self.exp_dir, 'epoch'))
        elif platform.system().lower() == 'windows' and os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch1_test_score.pkl')):
            os.system('del \"%s\epoch*.pkl\"' % os.path.join(self.exp_dir, 'epoch'))
        print('finish merge: ', self.exp_dir)

    def get_final_score(self):
        file = [kk for kk in os.listdir(os.path.join(self.exp_dir, 'epoch'))
                if re.search('allepo_\d+_test_score.pkl', kk) is not None]
        if not (len(file) == 1 and file[0][:6] == 'allepo'):
            print(os.path.join(self.exp_dir, 'epoch'), str(file))
            return None
        with open(os.path.join(self.exp_dir, 'epoch', file[0]), 'rb') as f:
            score_pred_df = pickle.load(f)
        score_pred_df = pd.DataFrame(score_pred_df[len(score_pred_df)])
        with open(os.path.join(self.data_dir, 'label.pkl'), 'rb') as f:
            label_all, sub_name_all = pickle.load(f)
        label_dict = dict(zip(sub_name_all, label_all))
        label_df = pd.DataFrame(label_dict, index=['true_label'])
        scores = pd.concat([score_pred_df, label_df], axis=0).dropna(axis=1)
        return scores

    def confusion_matrix(self, out_path):
        scores = self.get_final_score()
        cm = get_CM(scores.iloc[-1, :], scores.iloc[:-1, :])
        classes = ['0', 'n0'] if self.data_name.split('_')[2] == '0n0' else [str(kk) for kk in self.data_name.split('_')[2]]
        plot_confusion_matrix(self.data_name + '\n' + self.para_name, out_path, cm=cm, classes=classes)

    def plot_att(self, fig_size1=None, fig_size2=None, vmin=0, vmax=1, bar_ticks=np.arange(10) / 10):
        if fig_size1 is None:
            fig_size1 = [20, 80]
        if fig_size2 is None:
            fig_size2 = [20, 10]
        score_att = pd.read_csv(os.path.join(self.exp_dir, "all_sample_final_score_att.csv"), index_col=0)

        slc_roi = []
        for cc in range(len(self.coor)):
            i, j, k = self.coor[cc]
            slc_roi.append(int(self.related_roi_arr[i, j, k]))
        self.slc_roi = slc_roi  # every patch corresponding to a ROI index, '0' means unrelated ROI
        self.roi_cut = np.where(np.array(self.slc_roi) > 0)[0]
        print('how many patches are included in related ROIs: ', len(self.roi_cut))

        fig, ax = plt.subplots(figsize=(fig_size1[0], fig_size1[1]))
        xx = score_att.iloc[(self.num_class + 1):, :].values
        print('all att score range: ', xx.min(), xx.max())
        plt.imshow(xx, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_yticks(np.arange(len(self.slc_roi)))  # 设置y轴间隔
        ax.set_yticklabels(self.slc_roi)  # 设置y轴标签
        plt.colorbar(ticks=list(bar_ticks))
        plt.savefig(os.path.join(self.exp_dir, "all_sample_att_heatmap.png"), dpi=100, bbox_inches='tight')
        plt.show()

        # 截取子图
        fig, ax = plt.subplots(figsize=(fig_size2[0], fig_size2[1]))
        xx = score_att.iloc[self.roi_cut + 3, :].values
        print('ROI att score range: ', xx.min(), xx.max())
        plt.imshow(xx, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_yticks(np.arange(len(self.roi_cut)))  # 设置y轴间隔
        ax.set_yticklabels(np.array(self.slc_roi)[self.roi_cut])  # 设置y轴标签
        plt.colorbar(ticks=list(bar_ticks))
        plt.savefig(os.path.join(self.exp_dir, "all_sample_att_heatmap_cut.png"), dpi=100, bbox_inches='tight')
        plt.show()

    def cal_cover_score(self):
        score_att = pd.read_csv(os.path.join(self.exp_dir, "all_sample_final_score_att.csv"), index_col=0)
        att_ave = np.mean(score_att.iloc[(self.num_class + 1):, :].values, axis=1)

        att_mask_arr = np.zeros(self.atlas_arr.shape)
        width = int(self.patch_size / 2)
        for cc in range(len(self.coor)):
            i, j, k = self.coor[cc]
            att_mask_arr[i - width:i + width + 1, j - width:j + width + 1, k - width:k + width + 1] = att_ave[cc]
        att_mask_nii = nib.Nifti1Image(att_mask_arr, affine=self.mask_affine, header=self.mask_header, dtype=np.int8)
        nib.save(att_mask_nii, os.path.join(self.exp_dir, "select_att.nii.gz"))

        count = 0
        score = 0
        for cc in range(len(self.coor)):
            i, j, k = self.coor[cc]
            if self.related_roi_arr[i, j, k] > 0:
                count += 1
                score += att_ave[cc]
        assert len(self.roi_cut) == count
        print('score rate of patches which are covered in related ROIs: %.4f=%.4f/%.4f' % \
              (score / att_ave.sum(), score, att_ave.sum()))


if __name__ == '__main__':
    main()

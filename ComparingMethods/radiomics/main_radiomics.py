from __future__ import print_function
import os
import random
import yaml
import pickle
from shutil import copytree, ignore_patterns
import argparse
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from tools.utils import eval_metric_cl2, get_auc_cl2, get_CM, plot_confusion_matrix
from tools.feeder import Prepare


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--save_dir', default='./', help='the work folder for storing results')
    parser.add_argument('--data_dir', default='./')
    parser.add_argument('--config', default='./train_radiomics.yaml', help='path to the configuration file')
    parser.add_argument('--seed', default=1, type=int, help='seed for random')
    parser.add_argument('--split_seed', default=1, type=int)
    # parser.add_argument('--fold', default=0, type=int, help='0-4, fold idx for cross-validation')
    parser.add_argument('--Y_name', default='arising_2_0n0', type=str)
    parser.add_argument('--patch_size', default=5, type=int)
    parser.add_argument('--rate', default=0.5, type=float)
    parser.add_argument('--radio_path', default='./radiomics_features/radiomics_features_88.csv', type=str)

    return parser


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

    def save_arg(self):
        self.num_class = int(self.arg.Y_name.split('_')[1])
        self.data_name = '{s[0]}_pw{s[1]}_r{s[2]}'.format(
            s=[str(vars(self.arg)[kk]) for kk in ['Y_name', 'patch_size', 'rate']])
        self.arg.exp_name = 'radiomics_classification'
        self.work_dir = os.path.join(self.arg.save_dir, self.data_name, f'split{self.arg.split_seed}',
                                     f'seed{self.arg.seed}')
        os.makedirs(self.work_dir, exist_ok=True)
        self.data_path = os.path.join(self.arg.data_dir, self.data_name.split('_pw')[0], self.data_name)

        # copy all files
        pwd = os.path.dirname(os.path.realpath(__file__))
        copytree(pwd, os.path.join(self.work_dir, 'code'), symlinks=False, ignore=ignore_patterns('__pycache__'), dirs_exist_ok=True)
        arg_dict = vars(self.arg)
        with open(os.path.join(self.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def start(self):
        df = pd.read_csv(self.arg.radio_path)
        with open(os.path.join(self.data_path, 'label.pkl'), 'rb') as f:
            label_all, sub_name_all = pickle.load(f)
        df_use = pd.merge(pd.DataFrame({'sub_name': sub_name_all, 'label': label_all}), df, on='sub_name', how='left')

        scores = []
        for i in range(5):
            data_dict = self.load_data(df_use, fold=i)
            probs_dict = self.classic_clf(data_dict)
            scores.append(probs_dict)
        with open(os.path.join(self.work_dir, 'test_score.pkl'), 'wb') as f:
            pickle.dump(scores, f)

        evals_dict = {}
        clf_name = ['SVC', 'LogisticRegression', 'GaussianNB', 'KNeighbors', 'DecisionTree', 'RandomForest']
        ave = ''
        metrics = ['acc', 'pre', 'sen', 'spe', 'f1', 'auc']
        true_label = np.hstack([scores[ii]['true'] for ii in range(5)])

        for mm in clf_name:
            scores_5CV = []
            for fold in range(5):
                vv = scores[fold][mm]
                scores_5CV.append(vv)
            scores_5CV = np.vstack(scores_5CV)
            if self.num_class == 2:
                auc, _, _ = get_auc_cl2(true=true_label, prob=scores_5CV, softmax=False)
                evals = eval_metric_cl2(true=true_label, prob=scores_5CV)
                evals_dict.update({mm: [evals[kk] for kk in metrics[:-1]] + [auc]})
            else:
                auc, _, _ = get_auc(true=true_label, prob=scores_5CV, softmax=False)
                evals = eval_metric(true=true_label, prob=scores_5CV)
                evals_dict.update({mm: [evals[ave][kk] for kk in metrics[:-1]] + [auc[ave]]})
            cm = get_CM(true=true_label, prob=scores_5CV)
            classes = ['0', 'n0'] if self.data_name.split('_')[2] == '0n0' else [str(kk) for kk in self.data_name.split('_')[2]]
            plot_confusion_matrix(self.data_name + '\n' + mm, os.path.join(self.work_dir, mm + '.png'),
                                  cm=cm, classes=classes)

        evals_df = pd.DataFrame(evals_dict, index=[kk if kk[:3] == 'acc' else f'{kk}_{ave}' for kk in metrics])
        evals_df.to_csv(os.path.join(self.work_dir, f'finalscores_5CV_para_split{self.arg.split_seed}_seed{self.arg.seed}.csv'), index=True)

    def load_data(self, df_use, fold):
        data_dict = dict()
        for mode in ['train', 'test']:
            test_idx = np.loadtxt(os.path.join(self.data_path, f'test_idx_{self.arg.split_seed}', f'{fold}.txt')).astype(int)
            train_idx = np.array(list(set(range(len(df_use))) - set(test_idx)))
            idx = train_idx if mode == 'train' else test_idx

            data = df_use.iloc[idx, 3:].values
            label = df_use['label'][idx].values
            sample_name = df_use['sub_name'][idx].values

            data_dict.update({f'x_{mode}': data, f'y_{mode}': label, f'sub_{mode}': sample_name})

        return data_dict

    def classic_clf(self, data_dict):
        X_tr_tmp, X_te_tmp, Y_train, Y_test = \
            data_dict['x_train'], data_dict['x_test'], data_dict['y_train'], data_dict['y_test']
        probs_dict = {'true': data_dict['y_test']}

        # svm_clf = Pipeline([("scaler", StandardScaler()), ("svc_rbf", SVC(kernel="rbf", probability=True))])
        svm_clf = SVC(kernel="rbf", probability=True)
        svm_clf.fit(X_tr_tmp, Y_train)
        prob = svm_clf.predict_proba(X_te_tmp)
        probs_dict.update({'SVC': prob})

        oe = OrdinalEncoder()
        oe.fit(Y_train.reshape(-1, 1)).categories_
        Y_train0 = oe.fit_transform(Y_train.reshape(-1, 1))
        lr_clf = LogisticRegression()
        lr_clf.fit(X_tr_tmp, Y_train0.reshape(-1))
        prob = lr_clf.predict_proba(X_te_tmp)
        probs_dict.update({'LogisticRegression': prob})

        nb_clf = GaussianNB()
        nb_clf.fit(X_tr_tmp, Y_train)
        prob = nb_clf.predict_proba(X_te_tmp)
        probs_dict.update({'GaussianNB': prob})

        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_tr_tmp, Y_train)
        prob = knn_clf.predict_proba(X_te_tmp)
        probs_dict.update({'KNeighbors': prob})

        dt_clf = DecisionTreeClassifier(random_state=self.arg.seed, min_samples_leaf=10)
        dt_clf.fit(X_tr_tmp, Y_train)
        prob = dt_clf.predict_proba(X_te_tmp)
        probs_dict.update({'DecisionTree': prob})

        rf_clf = RandomForestClassifier(random_state=self.arg.seed)
        rf_clf.fit(X_tr_tmp, Y_train)
        prob = rf_clf.predict_proba(X_te_tmp)
        probs_dict.update({'RandomForest': prob})

        return probs_dict


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = get_parser()
    set_seed(parser.parse_args().seed)

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    processor = Processor(arg)
    processor.start()


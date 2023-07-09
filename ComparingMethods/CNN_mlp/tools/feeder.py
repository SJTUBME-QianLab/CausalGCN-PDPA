from __future__ import print_function
import numpy as np
import os
import pickle
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


class Feeder(Dataset):
    def __init__(self, fold, split_seed, out_dir, mode, debug=False, save=False):
        assert mode in ['train', 'test']
        self.fold_num = 5
        self.split_seed = split_seed
        self.fold = fold
        self.out_dir = out_dir
        self.mode = mode
        self.debug = debug
        self.save = save
        pp = Prepare(self.out_dir, self.fold_num)
        self.data, self.label, self.sample_name = pp.train_test_split(seed=split_seed, fold=fold, mode=mode)

        # self.coordinates = np.load(os.path.join(self.out_dir, 'coordinates.npy'))
        self.N, self.P, self.L, self.W, self.H = self.data.shape

        if self.debug:
            self.label = self.label[0:10]
            self.sample_name = self.sample_name[0:10]
            self.data = self.data[0:10]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]

        return sum(hit_top_k) * 1.0 / len(hit_top_k)


class Prepare:
    def __init__(self, data_path, fold_num=5):
        self.data_path = data_path
        self.fold_num = fold_num
        with open(os.path.join(data_path, 'label.pkl'), 'rb') as f:
            self.label_all, self.sub_name_all = pickle.load(f)
        self.data_all = np.load(os.path.join(data_path, 'data.npy'))
        self.coordinates = np.load(os.path.join(data_path, 'coordinates.npy'))

    def train_test_split(self, seed, fold, mode):
        skf = StratifiedKFold(n_splits=self.fold_num, shuffle=True, random_state=seed)
        split = skf.split(self.label_all, self.label_all)
        train_idx_list, test_idx_list = [], []
        save_idx_dir = os.path.join(self.data_path, 'test_idx_' + str(seed))
        os.makedirs(save_idx_dir, exist_ok=True)
        for i, (train, test) in enumerate(split):
            train_idx_list.append(train)
            test_idx_list.append(test)
            if os.path.isfile(os.path.join(save_idx_dir, '%d.txt' % i)):
                test_save = np.loadtxt(os.path.join(save_idx_dir, '%d.txt' % i)).astype(int)
                assert (test_save == test).all()
            else:
                np.savetxt(os.path.join(save_idx_dir, '%d.txt' % i), test, fmt="%d")

        if mode == 'train':
            data, label, sub = sub_data(self.data_all, self.label_all, self.sub_name_all, train_idx_list, fold)
        elif mode == 'test':
            data, label, sub = sub_data(self.data_all, self.label_all, self.sub_name_all, test_idx_list, fold)
        else:
            raise ValueError

        return data, label, sub


def sub_data(data, label, sub_name, idx_list, fold):
    data = data[idx_list[fold], ...]
    label = label[idx_list[fold]]
    sub_name = sub_name[idx_list[fold]]
    return data, label, sub_name


# if __name__ == '__main__':
#     import yaml
#     with open('./../train.yaml', 'r') as f:
#         default_arg = yaml.safe_load(f)
#     feader = Feeder(fold=0, **default_arg['train_feeder_args'])

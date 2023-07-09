from __future__ import print_function
import numpy as np
import os
import pickle
import platform
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


class Feeder(Dataset):
    def __init__(self, fold, split_seed, out_dir, mode, graph_arg, debug=False, save=False):
        assert mode in ['train', 'test']
        self.fold_num = 5
        self.split_seed = split_seed
        self.fold = fold
        self.out_dir = out_dir
        self.mode = mode
        self.graph_arg = graph_arg
        self.debug = debug
        self.save = save
        self.load_data()

    def load_data(self):
        data_file = os.path.join(self.out_dir, 'test_idx_' + str(self.split_seed),
                                 self.mode + str(self.fold) + '.npz')
        if os.path.isfile(data_file):
            dd = np.load(data_file, allow_pickle=True)
            self.data, self.label, self.sample_name = dd['data'], dd['label'], dd['sub_name']
        else:
            pp = Prepare(self.out_dir, self.fold_num)
            self.data, self.label, self.sample_name = pp.train_test_split(
                out_path=self.out_dir, seed=self.split_seed, fold=self.fold, mode=self.mode, save=self.save)

        self.coordinates = np.load(os.path.join(self.out_dir, 'coordinates.npy'))
        self.N, self.P, self.L, self.W, self.H = self.data.shape

        if self.debug:
            self.label = self.label[0:10]
            self.sample_name = self.sample_name[0:10]
            self.data = self.data[0:10]
            self.edge_list = self.edge_list[0:10]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        edges = self.edge_list[index].toarray()
        label = self.label[index]

        return data_numpy, edges, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]

        return sum(hit_top_k) * 1.0 / len(hit_top_k)


class Prepare:
    def __init__(self, data_path, fold_num):
        self.data_path = data_path
        self.fold_num = fold_num
        with open(os.path.join(self.data_path, 'label.pkl'), 'rb') as f:
            self.label_all, self.sub_name_all = pickle.load(f)
        self.data_all = np.load(os.path.join(self.data_path, 'data.npy'))
        self.coordinates = np.load(os.path.join(self.data_path, 'coordinates.npy'))

    def train_test_split(self, out_path, seed, fold, mode, save=False):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        split = skf.split(self.label_all, self.label_all)
        train_idx_list, test_idx_list = [], []
        save_idx_dir = os.path.join(out_path, 'test_idx_' + str(seed))
        os.makedirs(save_idx_dir, exist_ok=True)
        for i, (train, test) in enumerate(split):
            train_idx_list.append(train)
            test_idx_list.append(test)
            if os.path.isfile(os.path.join(save_idx_dir, '%d.txt' % i)):
                test_save = np.loadtxt(os.path.join(save_idx_dir, '%d.txt' % i)).astype(int)
                assert (test_save == test).all()
            else:
                np.savetxt(os.path.join(save_idx_dir, '%d.txt' % i), test, fmt="%d")

        if save and os.path.isfile(os.path.join(save_idx_dir, 'test4.npz')):
            for i in range(5):
                data, label, sub = self.sub_data(self.data_all, self.label_all, self.sub_name_all, train_idx_list, i)
                np.savez(os.path.join(save_idx_dir, 'train%d.npz' % i), data=data, label=label, sub_name=sub)

                data, label, sub = self.sub_data(self.data_all, self.label_all, self.sub_name_all, test_idx_list, i)
                np.savez(os.path.join(save_idx_dir, 'test%d.npz' % i), data=data, label=label, sub_name=sub)

        if mode == 'train':
            data, label, sub = self.sub_data(self.data_all, self.label_all, self.sub_name_all, train_idx_list, fold)
        elif mode == 'test':
            data, label, sub = self.sub_data(self.data_all, self.label_all, self.sub_name_all, test_idx_list, fold)
        else:
            raise ValueError

        return data, label, sub

    @staticmethod
    def sub_data(data, label, sub_name, idx_list, fold):
        data = data[idx_list[fold], ...]
        label = label[idx_list[fold]]
        sub_name = sub_name[idx_list[fold]]
        return data, label, sub_name


if __name__ == '__main__':
    import yaml
    with open('./../train.yaml', 'r') as f:
        default_arg = yaml.safe_load(f)
    feader = Feeder(fold=0, **default_arg['train_feeder_args'])

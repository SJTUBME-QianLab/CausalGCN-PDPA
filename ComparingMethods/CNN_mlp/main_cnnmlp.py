from __future__ import print_function
import os
import time
import yaml
import pickle
from shutil import copytree, ignore_patterns
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tools.utils import seed_torch, str2bool, str2list, get_emb_name, import_class
from settle_results import SettleResults
# # TF_ENABLE_ONEDNN_OPTS = 0
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--save_dir', default='./results', help='the work folder for storing results')
    parser.add_argument('--data_dir', default='./')
    parser.add_argument('--config', default='./train_cnnmlp.yaml', help='path to the configuration file')
    parser.add_argument('--seed', default=1, type=int, help='seed for random')
    parser.add_argument('--split_seed', default=1, type=int)
    parser.add_argument('--fold', default=0, type=int, help='0-4, fold idx for cross-validation')
    parser.add_argument('--Y_name', default='arising_2_0n0', type=str)
    parser.add_argument('--patch_size', default=5, type=int)
    parser.add_argument('--rate', default=0.5, type=float)

    # processor
    # parser.add_argument('--phase', default='cnn', help='must be train or test')
    parser.add_argument('--save_score', type=str2bool, default=True,
                        help='if ture, the classification score will be stored')

    # visualize and debug
    parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save_interval', type=int, default=5, help='the interval for storing models (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show_topk', type=int, default=[1], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='tools.feeder.Feeder', help='data loader will be used')
    parser.add_argument('--num_worker', type=int, default=0, help='the number of worker for data loader')
    parser.add_argument('--train_feeder_args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test_feeder_args', default=dict(), help='the arguments of data loader for test')

    # model
    # parser.add_argument('--graph', default=None, help='the model will be used')
    # parser.add_argument('--graph_args', default=dict(), help='the arguments of model')  # type=dict,
    # parser.add_argument('--model', default=None, help='the model will be used')
    # parser.add_argument('--model_args', default=dict(), help='the arguments of model')  # type=dict,
    # parser.add_argument('--weights', default=None, help='the weights for network initialization')
    # parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
    #                     help='the name of weights which will be ignored in the initialization')

    # CNN embedding
    parser.add_argument('--emb', default=None, help='the model will be used')
    parser.add_argument('--emb_args', default=dict(), help='the arguments of model')
    parser.add_argument('--cnn_pre_epoch', type=int, default=20)
    # parser.add_argument('--cnn_weights', default=None, help='the weights for CNN network initialization')

    # optimizer
    parser.add_argument('--base_lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch_size', type=int, default=50, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=50, help='test batch size')
    # parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    # parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    return parser


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.main_cnn_emb()

    def save_arg(self):
        self.num_class = int(self.arg.Y_name.split('_')[1])

        self.arg.emb_args = str2list(self.arg.emb_args, flag='deep')

        self.data_name = '{s[0]}_pw{s[1]}_r{s[2]}'.format(
            s=[str(vars(self.arg)[kk]) for kk in ['Y_name', 'patch_size', 'rate']])
        self.data_path = os.path.join(self.arg.data_dir, self.data_name.split('_pw')[0], self.data_name)
        if self.arg.emb_args['hidden2'] is None:
            emb_name = 'C{}K{}h0{}_lr{:g}e{:d}'.format(
                '.'.join([str(s) for s in self.arg.emb_args['feature_depth']]),
                '.'.join([str(s) for s in self.arg.emb_args['kernels']]),
                self.arg.emb_args['pool'],
                self.arg.base_lr, self.arg.cnn_pre_epoch
            )
        else:
            emb_name = 'C{}K{}h{}{}_lr{:g}e{:d}'.format(
                '.'.join([str(s) for s in self.arg.emb_args['feature_depth']]),
                '.'.join([str(s) for s in self.arg.emb_args['kernels']]),
                '.'.join([str(s) for s in self.arg.emb_args['hidden2']]),
                self.arg.emb_args['pool'],
                self.arg.base_lr, self.arg.cnn_pre_epoch
            )
        self.arg.exp_name = '__'.join([emb_name, self.arg.exp_name])
        self.model_name = 't{}__{}'.format(time.strftime('%Y%m%d%H%M%S'), self.arg.exp_name)
        self.work_dir = os.path.join(self.arg.save_dir, self.data_name, f'split{self.arg.split_seed}', f'seed{self.arg.seed}', f'fold{self.arg.fold}', self.model_name)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'epoch'), exist_ok=True)  # save pt, pkl
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device

        # copy all files
        pwd = os.path.dirname(os.path.realpath(__file__))
        copytree(pwd, os.path.join(self.work_dir, 'code'), symlinks=False, ignore=ignore_patterns('__pycache__'), dirs_exist_ok=True)
        arg_dict = vars(self.arg)
        with open(os.path.join(self.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def load_data(self):
        self.print_log('(CNN prepared) Load data.')
        Feeder = import_class(self.arg.feeder)
        train_set = Feeder(fold=self.arg.fold, split_seed=self.arg.split_seed,
                           out_dir=self.data_path, mode='train', **self.arg.train_feeder_args)
        test_set = Feeder(fold=self.arg.fold, split_seed=self.arg.split_seed,
                          out_dir=self.data_path, mode='test', **self.arg.test_feeder_args)
        self.data_loader = dict()
        self.data_loader['train'] = DataLoader(
            dataset=train_set, batch_size=self.arg.batch_size, num_workers=self.arg.num_worker,
            shuffle=True, drop_last=True)
        self.data_loader['train_eval'] = DataLoader(
            dataset=train_set, batch_size=self.arg.batch_size, num_workers=self.arg.num_worker,
            shuffle=False, drop_last=False)
        self.data_loader['test'] = DataLoader(
            dataset=test_set, batch_size=self.arg.test_batch_size, num_workers=self.arg.num_worker,
            shuffle=False, drop_last=False)
        assert self.data_loader['test'].dataset.L == self.arg.patch_size

    def main_cnn_emb(self):
        self.load_data()
        self.embedding = import_class(self.arg.emb)(
                num_class=self.num_class, patch_num=self.data_loader['test'].dataset.P,
                **self.arg.emb_args
            ).cuda(self.output_device)
        if self.arg.optimizer == 'SGD':
            self.opt_emb = optim.SGD(self.embedding.parameters(),
                                     lr=self.arg.base_lr, weight_decay=self.arg.weight_decay,
                                     momentum=0.9, nesterov=self.arg.nesterov)
        elif self.arg.optimizer == 'Adam':
            self.opt_emb = optim.Adam(self.embedding.parameters(),
                                      lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.lr_cnn = self.arg.base_lr
        self.lr_scheduler_cnn = ReduceLROnPlateau(self.opt_emb, mode='min', factor=0.1, patience=10, verbose=True,
                                                  threshold=1e-4, threshold_mode='rel', cooldown=0)
        self.cnn_train_writer = SummaryWriter(os.path.join(self.work_dir, 'cnn_train'), 'cnn_train')
        self.cnn_test_writer = SummaryWriter(os.path.join(self.work_dir, 'cnn_test'), 'cnn_test')
        self.global_step_cnn = 0
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        for epoch in range(self.arg.cnn_pre_epoch):
            self.embedding.train()
            self.train_cnn(epoch)
            self.embedding.eval()
            with torch.no_grad():
                self.eval_cnn(epoch, loader_name=['train_eval', 'test'], save_score=self.arg.save_score)
        # os.makedirs(os.path.split(self.emb_dir)[0], exist_ok=True)
        # np.savez(self.emb_dir, train_x=data_dict['train_eval_x'], test_x=data_dict['test_x'],
        #          train_pred=data_dict['train_eval_y'], test_pred=data_dict['test_y'])

        ss = SettleResults(self.data_loader['test'].dataset.out_dir, self.work_dir, self.arg.exp_name)
        ss.concat_trend_scores(num_epoch=self.arg.cnn_pre_epoch,
                               metrics=['acc', 'pre', 'sen', 'spe', 'f1'], phase='test', ave='')
        ss.merge_pkl(num_epoch=self.arg.cnn_pre_epoch, type_list=['train_eval_score', 'test_score'])
        ss.confusion_matrix(out_path=os.path.join(self.work_dir, 'CM.png'))
        print('finish: ', self.work_dir)
        
    def train_cnn(self, epoch):
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        try:
            process = tqdm(loader, ncols=50)
        except KeyboardInterrupt:
            process.close()
            raise
        process.close()

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step_cnn += 1
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)  # [12, 200, 5, 5, 5]
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            if epoch == 0 and batch_idx == 0:
                self.flop_counter(self.embedding, input=(data, ))
            pred = self.embedding(data)

            loss = self.loss(pred, label)
            self.opt_emb.zero_grad()
            loss.backward()
            self.opt_emb.step()
            loss_value.append(loss.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(pred.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.cnn_train_writer.add_scalar(self.model_name + '/cnn_acc', acc, self.global_step_cnn)
            self.cnn_train_writer.add_scalar(self.model_name + '/cnn_loss', loss.item(), self.global_step_cnn)
            self.lr_cnn = self.opt_emb.param_groups[0]['lr']
            self.cnn_train_writer.add_scalar(self.model_name + '/lr', self.lr_cnn, self.global_step_cnn)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}, [Evaluate]{statistics}'.format(**proportion))

    def eval_cnn(self, epoch, loader_name=['test'], save_score=False):
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        data_dict = {}
        for ln in loader_name:
            loss_value = []
            score_dict = {}
            try:
                process = tqdm(self.data_loader[ln], ncols=50)
            except KeyboardInterrupt:
                process.close()
                raise
            process.close()

            for batch_idx, (data, label, index) in enumerate(process):
                data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                label = Variable(label.long().cuda(self.output_device), requires_grad=False)

                pred = self.embedding(data)
                loss = self.loss(pred, label)
                loss_value.append(loss.item())
                sub_list = self.data_loader[ln].dataset.sample_name[index] if len(index) > 1 \
                    else [self.data_loader[ln].dataset.sample_name[index]]
                score_dict.update(dict(zip(sub_list, pred.data.cpu().numpy())))

            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(np.array(list(score_dict.values())), 1)
            self.print_log(f'ln: {ln}, Accuracy: {accuracy}, model: {self.model_name}')
            self.lr_scheduler_cnn.step(loss)
            self.cnn_test_writer.add_scalar(self.model_name + '/cnn_acc', accuracy, self.global_step_cnn)
            self.cnn_test_writer.add_scalar(self.model_name + '/cnn_loss', loss, self.global_step_cnn)

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(np.array(list(score_dict.values())), k)))
            if save_score:
                with open(os.path.join(self.work_dir, 'epoch', 'epoch{}_{}_score.pkl'.format(epoch + 1, ln)), 'wb') as f:
                    pickle.dump(score_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open(os.path.join(self.work_dir, 'log.txt'), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def flop_counter(self, model, input):
        """
        https://zhuanlan.zhihu.com/p/460113408
        :param model:
        :param input:
        :return:
        """
        from thop import profile
        from thop import clever_format
        macs, params = profile(model, inputs=input, verbose=False)
        self.print_log(f"macs = {macs}")
        self.print_log(f"params = {params}")

        macs, params = clever_format([macs, params], "%.3f")  # clever_format
        self.print_log(f"Macs={macs}")
        self.print_log(f"Params={params}")

        total = sum([param.nelement() for param in model.parameters()])
        self.print_log(f"total={total}")


if __name__ == '__main__':
    parser = get_parser()

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
    seed_torch(arg.seed)
    processor = Processor(arg)


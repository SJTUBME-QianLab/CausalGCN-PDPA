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
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from tools.utils import seed_torch, str2bool, str2list, get_graph_name, get_emb_name, import_class
from settle_results import SettleResults
TF_ENABLE_ONEDNN_OPTS = 0
# torch.use_deterministic_algorithms(True)
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from net.braingnn import Network


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--save_dir', default='./results', help='the work folder for storing results')
    parser.add_argument('--data_dir', default='./')
    parser.add_argument('--config', default='./train_braingnn.yaml', help='path to the configuration file')
    parser.add_argument('--seed', default=1, type=int, help='seed for random')
    parser.add_argument('--split_seed', default=1, type=int)
    parser.add_argument('--fold', default=0, type=int, help='0-4, fold idx for cross-validation')
    parser.add_argument('--Y_name', default='arising_2_0n0', type=str)
    parser.add_argument('--patch_size', default=5, type=int)
    parser.add_argument('--rate', default=0.5, type=float)

    # processor
    # parser.add_argument('--phase', default='train', help='must be train or test')
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
    parser.add_argument('--graph_args', default=dict(), help='the arguments of model')  # type=dict,
    # parser.add_argument('--model', default=None, help='the model will be used')
    # parser.add_argument('--model_args', default=dict(), help='the arguments of model')  # type=dict,
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')

    # CNN embedding
    # parser.add_argument('--emb', default=None, help='the model will be used')
    parser.add_argument('--emb_args', default=dict(), help='the arguments of model')  # type=dict,
    parser.add_argument('--cnn_pre_epoch', type=int, default=20)
    # parser.add_argument('--cnn_weights', default=None, help='the weights for CNN network initialization')

    # optimizer
    parser.add_argument('--base_lr', type=float, default=0.001, help='initial learning rate')  # 初始学习率
    parser.add_argument('--step', type=int, default=[], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')  # 优化器降低学习率的epoch
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')  # Momentum优化算法的一种改进
    parser.add_argument('--batch_size', type=int, default=50, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=50, help='test batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')

    parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
    parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
    parser.add_argument('--lamb1', type=float, default=0.1, help='s1 unit regularization')
    parser.add_argument('--lamb2', type=float, default=0.1, help='s2 unit regularization')
    parser.add_argument('--lamb3', type=float, default=0.1, help='s1 entropy regularization')
    parser.add_argument('--lamb4', type=float, default=0.1, help='s2 entropy regularization')
    parser.add_argument('--lamb5', type=float, default=0.1, help='s1 consistence regularization')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
    # parser.add_argument('--indim', type=int, default=200, help='feature dim')
    # parser.add_argument('--nroi', type=int, default=200, help='num of ROIs')
    return parser


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.load_data()
        self.train_writer = SummaryWriter(os.path.join(self.work_dir, 'train'), 'train')
        self.val_writer = SummaryWriter(os.path.join(self.work_dir, 'train_val'), 'train_val')
        self.test_writer = SummaryWriter(os.path.join(self.work_dir, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def save_arg(self):
        self.num_class = int(self.arg.Y_name.split('_')[1])

        self.arg.emb_args = str2list(self.arg.emb_args, flag='deep')
        self.arg.graph_args = str2list(self.arg.graph_args, flag='simple')

        self.data_name = '{s[0]}_pw{s[1]}_r{s[2]}'.format(
            s=[str(vars(self.arg)[kk]) for kk in ['Y_name', 'patch_size', 'rate']])
        self.data_path = os.path.join(self.arg.data_dir, self.data_name.split('_pw')[0], self.data_name)
        self.emb_name = get_emb_name(epoch=self.arg.cnn_pre_epoch, seed=self.arg.split_seed, fold=self.arg.fold,
                                     **self.arg.emb_args)
        self.graph_name = get_graph_name(**self.arg.graph_args)
        netw = 'cls{:.2f}_uni{:.2f}_ent{:.2f}_cons{:.2f}_r{}'.format(
            self.arg.lamb0, self.arg.lamb1, self.arg.lamb3, self.arg.lamb5, self.arg.ratio
        )
        self.arg.exp_name = '__'.join([netw, self.arg.exp_name])

        self.model_name = 't{}__{}'.format(time.strftime('%Y%m%d%H%M%S'), self.arg.exp_name)
        self.work_dir = os.path.join(self.arg.save_dir, self.data_name, f'split{self.arg.split_seed}',
                                     f'seed{self.arg.seed}', f'fold{self.arg.fold}', self.model_name)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'epoch'), exist_ok=True)  # save pt, pkl
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.print_log(',\t'.join([self.emb_name, self.graph_name, netw, self.arg.exp_name]))

        # copy all files
        pwd = os.path.dirname(os.path.realpath(__file__))
        copytree(pwd, os.path.join(self.work_dir, 'code'), symlinks=False, ignore=ignore_patterns('__pycache__'), dirs_exist_ok=True)
        arg_dict = vars(self.arg)
        with open(os.path.join(self.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

        self.EPS = 1e-10

    def load_data(self):
        self.print_log('Load data.')
        Feeder = import_class(self.arg.feeder)
        train_set = Feeder(emb_name=self.emb_name, fold=self.arg.fold, split_seed=self.arg.split_seed,
                           out_dir=self.data_path, mode='train', graph_arg=self.arg.graph_args, **self.arg.train_feeder_args, )
        test_set = Feeder(emb_name=self.emb_name, fold=self.arg.fold, split_seed=self.arg.split_seed,
                          out_dir=self.data_path, mode='test', graph_arg=self.arg.graph_args, **self.arg.test_feeder_args)

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

    def load_model(self):
        # self.CELoss = nn.CrossEntropyLoss(reduction="mean")
        # self.CElosses = nn.CrossEntropyLoss(reduction="none")
        self.model = Network(
            indim=self.data_loader['test'].dataset.dim,
            R=self.data_loader['test'].dataset.P,
            ratio=self.arg.ratio, nclass=self.num_class,
        ).cuda(self.output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.arg.base_lr, weight_decay=self.arg.weight_decay,
                                       momentum=0.9, nesterov=self.arg.nesterov)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.lr_scheduler = StepLR(self.optimizer, step_size=self.arg.stepsize, gamma=self.arg.gamma)

    def start(self):
        # self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            # save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
            save_model = False
            self.train_mode()
            self.train(epoch, save_model=save_model)
            self.val_mode()
            with torch.no_grad():
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['train_eval', 'test'])

        print('best accuracy: ', self.best_acc, ' model_name: ', self.model_name)

        # settle results
        ss = SettleResults(self.data_loader['test'].dataset.out_dir, self.work_dir, self.arg.exp_name)
        ss.concat_trend_scores(num_epoch=self.arg.num_epoch, start_epoch=self.arg.start_epoch,
                               metrics=['acc', 'pre', 'sen', 'spe', 'f1'], phase='test', ave='')
        ss.merge_pkl(num_epoch=self.arg.num_epoch, start_epoch=self.arg.start_epoch,
                     type_list=['test_score', 'train_eval_score'])
        ss.confusion_matrix(out_path=os.path.join(self.work_dir, 'CM.png'))
        print('finish: ', self.work_dir)

    def train(self, epoch, save_model=False):
        self.print_log('Training epoch: {}'.format(epoch + 1))
        for param_group in self.optimizer.param_groups:
            self.print_log(f"LR, {param_group['lr']}")

        loader = self.data_loader['train']
        self.train_writer.add_scalar(self.model_name + '/epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        try:
            process = tqdm(loader, ncols=50)
        except KeyboardInterrupt:
            process.close()
            raise
        process.close()

        loss_value = []
        scores, true = [], []
        s1_list = []  # L_TPK
        s2_list = []  # L_GLC
        for batch_idx, (data, edges, label, index) in enumerate(process):
            self.global_step += 1
            x_node, edge_index, edge_attr, batch, pos, label = self.converse2tensor(data, edges, label)
            timer['dataloader'] += self.split_time()

            output, w1, w2, s1, s2 = self.model(x_node, edge_index, batch.view(-1), edge_attr, pos)
            timer['model'] += self.split_time()

            s1_list.append(s1.view(-1).detach().cpu().numpy())
            s2_list.append(s2.view(-1).detach().cpu().numpy())
            loss_c = F.nll_loss(output, label)

            loss_p1 = (torch.norm(w1, p=2) - 1) ** 2  # L_unit, Eq.7
            loss_p2 = (torch.norm(w2, p=2) - 1) ** 2
            loss_tpk1 = self.topk_loss(s1, self.arg.ratio)  # Eq.9
            loss_tpk2 = self.topk_loss(s2, self.arg.ratio)
            loss_consist = 0
            for c in range(self.num_class):
                loss_consist += self.consist_loss(s1[label == c])  # Eq.8
            loss = self.arg.lamb0 * loss_c + self.arg.lamb1 * loss_p1 + self.arg.lamb2 * loss_p2 \
                   + self.arg.lamb3 * loss_tpk1 + self.arg.lamb4 * loss_tpk2 + self.arg.lamb5 * loss_consist
            self.train_writer.add_scalar(self.model_name + '/classification_loss', loss_c, self.global_step)
            self.train_writer.add_scalar(self.model_name + '/unit_loss1', loss_p1, self.global_step)
            self.train_writer.add_scalar(self.model_name + '/unit_loss2', loss_p2, self.global_step)
            self.train_writer.add_scalar(self.model_name + '/TopK_loss1', loss_tpk1, self.global_step)
            self.train_writer.add_scalar(self.model_name + '/TopK_loss2', loss_tpk2, self.global_step)
            self.train_writer.add_scalar(self.model_name + '/GCL_loss', loss_consist, self.global_step)
            self.train_writer.add_scalar(self.model_name + '/loss', loss.item(), self.global_step)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            # loss_all += loss.item() * len(data)
            loss_value.append(loss.item())

            true.extend(self.data_loader['train'].dataset.label[index])
            scores.extend(np.argmax(output.data.cpu().numpy(), axis=1))

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar(self.model_name + '/acc', acc, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar(self.model_name + '/lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # accuracy = np.mean(np.array(true) == np.array(scores))
        # self.val_writer.add_scalar(self.model_name + '/acc', accuracy, self.global_step)

        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}, [Evaluate]{statistics}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(self.work_dir, 'epoch',  'epoch-' + str(epoch+1) + '.pt'))

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_dict = {}
            causal_matrix = {}

            try:
                process = tqdm(self.data_loader[ln], ncols=50)
            except KeyboardInterrupt:
                process.close()
                raise
            process.close()

            for batch_idx, (data, edges, label, index) in enumerate(process):
                x_node, edge_index, edge_attr, batch, pos, label = self.converse2tensor(data, edges, label)

                output, w1, w2, s1, s2 = self.model(x_node, edge_index, batch.view(-1), edge_attr, pos)

                loss = F.nll_loss(output, label)
                loss_value.append(loss.item())
                sub_list = self.data_loader[ln].dataset.sample_name[index] if len(index) > 1 \
                    else [self.data_loader[ln].dataset.sample_name[index]]
                score_dict.update(dict(zip(sub_list, output.data.cpu().numpy())))

            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(np.array(list(score_dict.values())), 1)
            if accuracy > self.best_acc and ln == 'test':
                self.best_acc = accuracy
            self.print_log(f'ln: {ln}, Accuracy: {accuracy}, model: {self.model_name}')
            if ln == 'train_eval':
                self.val_writer.add_scalar(self.model_name + '/loss', loss, self.global_step)
                self.val_writer.add_scalar(self.model_name + '/acc', accuracy, self.global_step)
            elif ln == 'test':
                # self.lr_scheduler.step(loss)
                self.test_writer.add_scalar(self.model_name + '/loss', loss, self.global_step)
                self.test_writer.add_scalar(self.model_name + '/acc', accuracy, self.global_step)

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(np.array(list(score_dict.values())), k)))

            if save_score:
                with open(os.path.join(self.work_dir, 'epoch', 'epoch{}_{}_score.pkl'.format(epoch + 1, ln)), 'wb') as f:
                    pickle.dump(score_dict, f)

    def converse2tensor(self, data, edges, label):
        B, P = data.shape[:2]
        batch = torch.cat([torch.ones(P) * kk for kk in range(B)], dim=0).long().cuda(self.output_device)
        data = Variable(data.float().cuda(self.output_device), requires_grad=False)
        edges = Variable(edges.float().cuda(self.output_device), requires_grad=False)
        label = Variable(label.long().cuda(self.output_device), requires_grad=False)
        pos = torch.cat([torch.eye(P)] * B, dim=0).to(self.output_device)

        edge_B_index = torch.LongTensor([[], []]).cuda(self.output_device)
        edge_B_attr = torch.tensor([]).cuda(self.output_device)
        for i, edge_square in enumerate(edges):
            edge_square = edge_square.to_sparse()
            edge_index, edge_attr = edge_square.indices(), edge_square.values()
            edge_B_index = torch.cat([edge_B_index, edge_index + i * P], dim=-1)
            edge_B_attr = torch.cat([edge_B_attr, edge_attr], dim=-1)
        return data.reshape(B*P, data.shape[-1]), edge_B_index, edge_B_attr, batch, pos, label

    def train_mode(self):
        self.model.train()

    def val_mode(self):
        self.model.eval()

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

    # ############################## Define Other Loss Functions ########################################
    def topk_loss(self, s, ratio):  # eq.9
        if ratio > 0.5:
            ratio = 1 - ratio
        s = s.sort(dim=1).values
        res = -torch.log(s[:, -int(s.size(1) * ratio):] + self.EPS).mean() - torch.log(
            1 - s[:, :int(s.size(1) * ratio)] + self.EPS).mean()
        return res

    def consist_loss(self, s):
        if len(s) == 0:
            return 0
        s = torch.sigmoid(s)
        W = torch.ones(s.shape[0], s.shape[0])
        D = torch.eye(s.shape[0]) * torch.sum(W, dim=1)
        L = D - W
        L = L.to(self.output_device)
        res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])  # S^T * L * S 矩阵乘法
        return res


if __name__ == '__main__':
    parser = get_parser()
    seed_torch(parser.parse_args().seed)

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


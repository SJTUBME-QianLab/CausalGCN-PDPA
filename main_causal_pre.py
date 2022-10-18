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

from tools.losses_define import *
from tools.utils import seed_torch, str2bool, str2list, get_graph_name, get_emb_name, import_class
from settle_results import SettleResults
TF_ENABLE_ONEDNN_OPTS = 0
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--save_dir', default='./results', help='the work folder for storing results')
    parser.add_argument('--data_dir', default='./')
    parser.add_argument('--config', default='./train_causal_pre.yaml', help='path to the configuration file')
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
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')  # type=dict,
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')

    # CNN embedding
    # parser.add_argument('--emb', default=None, help='the model will be used')
    parser.add_argument('--emb_args', default=dict(), help='the arguments of model')  # type=dict,
    parser.add_argument('--cnn_pre_epoch', type=int, default=20)
    # parser.add_argument('--cnn_weights', default=None, help='the weights for CNN network initialization')

    # hyper-parameters
    parser.add_argument('--loss_kind', type=str, default='own_yclose_l1_l1')
    parser.add_argument('--Laux', type=float, default=0)
    parser.add_argument('--L1', type=float, default=0)
    parser.add_argument('--L2', type=float, default=0)

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
        self.arg.model_args = str2list(self.arg.model_args, flag='deep')

        self.data_name = '{s[0]}_pw{s[1]}_r{s[2]}'.format(
            s=[str(vars(self.arg)[kk]) for kk in ['Y_name', 'patch_size', 'rate']])
        self.data_path = os.path.join(self.arg.data_dir, self.data_name.split('_pw')[0], self.data_name)
        self.emb_name = get_emb_name(epoch=self.arg.cnn_pre_epoch, seed=self.arg.split_seed, fold=self.arg.fold,
                                     **self.arg.emb_args)
        self.graph_name = get_graph_name(**self.arg.graph_args)
        netw = 'G{}_r{:.2f}'.format(
            '.'.join([str(s) for s in self.arg.model_args['hidden1'] + self.arg.model_args['hidden2']]),
            self.arg.model_args['ratio']
        )
        loss_name = '{}_La{:.4f}L1{:.4f}L2{:.4f}'.format(
            self.arg.loss_kind, self.arg.Laux, self.arg.L1, self.arg.L2
        )
        self.arg.exp_name = '__'.join([netw, loss_name, self.arg.exp_name])

        self.model_name = 't{}__{}'.format(time.strftime('%Y%m%d%H%M%S'), self.arg.exp_name)
        self.work_dir = os.path.join(self.arg.save_dir, self.data_name, f'split{self.arg.split_seed}',
                                     f'seed{self.arg.seed}', f'fold{self.arg.fold}', self.model_name)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'epoch'), exist_ok=True)  # save pt, pkl
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.print_log(',\t'.join([self.emb_name, self.graph_name, netw, loss_name, self.arg.exp_name]))

        # copy all files
        pwd = os.path.dirname(os.path.realpath(__file__))
        copytree(pwd, os.path.join(self.work_dir, 'code'), symlinks=False, ignore=ignore_patterns('__pycache__'), dirs_exist_ok=True)
        arg_dict = vars(self.arg)
        with open(os.path.join(self.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)  # 在结果目录保存一份运行时的参数配置文件

    def load_data(self):  # 加载数据
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

    def load_model(self):  # 加载模型
        self.CELoss = nn.CrossEntropyLoss(reduction="mean")
        self.CElosses = nn.CrossEntropyLoss(reduction="none")
        network = import_class(self.arg.model)
        self.causal_assign = network.CausalScore(
            dim=self.data_loader['train'].dataset.dim,
            hidden1=self.arg.model_args['hidden1'],
            ratio=self.arg.model_args['ratio'], norm=self.arg.graph_args['adj_norm']
        ).cuda(self.output_device)
        self.model = network.PredictionNet(
            in_features=self.arg.model_args['hidden1'][-1], num_class=self.num_class,
            hidden2=self.arg.model_args['hidden2']).cuda(self.output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(list(self.model.parameters()) + list(self.causal_assign.parameters()),
                                       lr=self.arg.base_lr, weight_decay=self.arg.weight_decay,
                                       momentum=0.9, nesterov=self.arg.nesterov)
            self.conf_opt = optim.SGD(self.model.conf_mlp.parameters(),
                                      lr=self.arg.base_lr, weight_decay=self.arg.weight_decay,
                                      momentum=0.9, nesterov=self.arg.nesterov)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.causal_assign.parameters()),
                                        lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
            self.conf_opt = optim.Adam(self.model.conf_mlp.parameters(),
                                       lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel', cooldown=0)  # 动态学习率缩减
        self.lr_scheduler_re = ReduceLROnPlateau(self.conf_opt, mode='min', factor=0.1, patience=10, verbose=True,
                                                 threshold=1e-4, threshold_mode='rel', cooldown=0)  # 动态学习率缩减

    def start(self):
        # self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        # self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
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
        for batch_idx, (data, edges, label, index) in enumerate(process):
            self.global_step += 1
            aux_loss, stable_loss, consist_loss, patch_loss = 0, 0, 0, 0
            # 加载数据和标签
            x_node, edge, label = self.converse2tensor(data, edges, label)
            timer['dataloader'] += self.split_time()

            (causal_x, causal_edge), (conf_x, conf_edge), node_score \
                = self.causal_assign(x_node, edge)
            causal_rep = self.model.get_graph_rep(x=causal_x, edge=causal_edge)  # 图编码器
            causal_out = self.model.get_causal_pred(causal_rep)  # 分类器
            conf_rep = self.model.get_graph_rep(x=conf_x, edge=conf_edge).detach()  # 图编码器 不回传
            conf_out = self.model.get_conf_pred(conf_rep)  # 另一个分类器

            timer['model'] += self.split_time()

            causal_loss = self.CELoss(causal_out, label)  # mean_j{L(y_j^c, y_j)}
            conf_loss = self.CELoss(conf_out, label)

            clf_loss, inv_loss, inv_type, con_loss = self.arg.loss_kind.split('_')

            # 因果（非因果辅助）的分类性能
            if self.arg.Laux > 0:
                aux_loss += custom_clf(label=label, model=self.model,
                                       rep_causal=causal_rep, rep_related=conf_rep, clf_loss=clf_loss)
                self.train_writer.add_scalar(self.model_name + '/loss_aux', aux_loss.item(), self.global_step)

            # 因果平稳性
            if self.arg.L1 > 0:
                stable_loss += custom_stable(label=label, model=self.model,
                                             rep_causal=causal_rep, rep_related=conf_rep,
                                             inv_loss=inv_loss, inv_type=inv_type)
                self.train_writer.add_scalar(self.model_name + '/loss_stable', stable_loss.item(), self.global_step)

            # 分配矩阵一致性
            if self.arg.L2 > 0:
                for c in range(self.num_class):
                    consist_loss += custom_consist(node_score[label == c], dist_type=con_loss)
                self.train_writer.add_scalar(self.model_name + '/loss_consist', consist_loss.item(), self.global_step)

            loss_sum = causal_loss + self.arg.Laux * aux_loss + self.arg.L1 * stable_loss + self.arg.L2 * consist_loss \

            self.conf_opt.zero_grad()
            conf_loss.backward()
            self.conf_opt.step()

            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()
            loss_value.append(causal_loss.item())

            true.extend(self.data_loader['train'].dataset.label[index])
            scores.extend(np.argmax(causal_out.data.cpu().numpy(), axis=1))

            value, predict_label = torch.max(causal_out.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar(self.model_name + '/acc', acc, self.global_step)
            self.train_writer.add_scalar(self.model_name + '/loss_all', loss_sum.item(), self.global_step)
            self.train_writer.add_scalar(self.model_name + '/loss_causal', causal_loss.item(), self.global_step)
            self.train_writer.add_scalar(self.model_name + '/loss_conf', conf_loss.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar(self.model_name + '/lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # accuracy = np.mean(np.array(true) == np.array(scores))
        # self.val_writer.add_scalar(self.model_name + '/acc', accuracy, self.global_step)

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
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value, loss_value2 = [], []
            score_dict, score_dictr, emb_all = {}, {}, {}
            causal_matrix = {}

            step = 0
            try:
                process = tqdm(self.data_loader[ln], ncols=50)
            except KeyboardInterrupt:
                process.close()
                raise
            process.close()

            for batch_idx, (data, edges, label, index) in enumerate(process):
                x_node, edge, label = self.converse2tensor(data, edges, label)

                (causal_x, causal_edge), (conf_x, conf_edge), node_score \
                    = self.causal_assign(x_node, edge)
                causal_rep = self.model.get_graph_rep(x=causal_x, edge=causal_edge)  # 图编码器
                causal_out = self.model.get_causal_pred(causal_rep)  # 分类器
                conf_rep = self.model.get_graph_rep(x=conf_x, edge=conf_edge).detach()  # 图编码器 不回传
                conf_out = self.model.get_conf_pred(conf_rep)  # 另一个分类器

                causal_loss = self.CELoss(causal_out, label)
                loss_value.append(causal_loss.item())
                conf_loss = self.CELoss(conf_out, label)
                loss_value2.append(conf_loss.item())
                sub_list = self.data_loader[ln].dataset.sample_name[index] if len(index) > 1 \
                    else [self.data_loader[ln].dataset.sample_name[index]]
                score_dict.update(dict(zip(sub_list, causal_out.data.cpu().numpy())))
                score_dictr.update(dict(zip(sub_list, conf_out.data.cpu().numpy())))
                causal_matrix.update(dict(zip(sub_list, node_score.data.cpu().numpy())))

                _, predict_label = torch.max(causal_out.data, 1)
                step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(np.array(list(score_dict.values())), 1)
            accuracyr = self.data_loader[ln].dataset.top_k(np.array(list(score_dictr.values())), 1)
            if accuracy > self.best_acc and ln == 'test':
                self.best_acc = accuracy
            self.print_log(f'ln: {ln}, Accuracy: {accuracy}, model: {self.model_name}')
            if ln == 'train_eval':
                self.val_writer.add_scalar(self.model_name + '/causal_loss', loss, self.global_step)
                self.val_writer.add_scalar(self.model_name + '/acc_causal', accuracy, self.global_step)
                self.val_writer.add_scalar(self.model_name + '/acc_conf', accuracyr, self.global_step)
            elif ln == 'test':
                self.lr_scheduler.step(loss)
                self.lr_scheduler_re.step(np.mean(loss_value2))
                self.test_writer.add_scalar(self.model_name + '/causal_loss', loss, self.global_step)
                self.test_writer.add_scalar(self.model_name + '/acc_causal', accuracy, self.global_step)
                self.test_writer.add_scalar(self.model_name + '/acc_conf', accuracyr, self.global_step)

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(np.array(list(score_dict.values())), k)))

            if epoch == (self.arg.num_epoch - 1):
                with open(os.path.join(self.work_dir, 'epoch', 'final_{}_Fassign.pkl'.format(ln)), 'wb') as f:
                    pickle.dump(causal_matrix, f)
            if save_score:
                with open(os.path.join(self.work_dir, 'epoch', 'epoch{}_{}_score.pkl'.format(epoch + 1, ln)), 'wb') as f:
                    pickle.dump(score_dict, f)

    def converse2tensor(self, data, edges, label):
        data = Variable(data.float().cuda(self.output_device), requires_grad=False)
        all_edge = Variable(edges.float().cuda(self.output_device), requires_grad=False)
        label = Variable(label.long().cuda(self.output_device), requires_grad=False)
        return data, all_edge, label

    def train_mode(self):
        self.model.train()
        self.causal_assign.train()

    def val_mode(self):
        self.model.eval()
        self.causal_assign.eval()

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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .GCN_layer import GCN as GraphConv_ordi
from tools.construct_graph_simple import normalize_digraph


class CausalScore(nn.Module):

    def __init__(self, dim, hidden1, ratio, norm):
        super(CausalScore, self).__init__()

        self.dim = dim
        self.hidden1 = hidden1
        self.convs1 = nn.ModuleList()
        # self.batch_norms1 = nn.ModuleList()
        for i in range(len(self.hidden1)):
            if i == 0:
                conv = GraphConv_ordi(in_features=self.dim, out_features=self.hidden1[0])
            else:
                conv = GraphConv_ordi(in_features=self.hidden1[i - 1], out_features=self.hidden1[i])
            self.convs1.append(conv)
            # self.batch_norms1.append(nn.BatchNorm1d(self.hidden1[i]))
        self.convs1.append(GraphConv_ordi(in_features=self.hidden1[-1], out_features=1))
        self.ratio = ratio
        self.norm = norm

    def forward(self, x, edge):
        node_score, x_1 = self.get_score(x, edge)
        (causal_x, causal_edge), (conf_x, conf_edge) = \
            split_graph_score_softmax(x_1, edge, node_score, self.ratio, self.norm)
        return (causal_x, causal_edge), (conf_x, conf_edge), node_score

    def get_score(self, x, edge):
        # for conv, bn in zip(self.convs1, self.batch_norms1):
        #     x = conv(x, edge)  # GraphConv
        #     x = F.relu(bn(x))
        for conv in self.convs1[:-1]:
            x = conv(x, edge)  # GraphConv
            x = F.relu(x)
        node_score = self.convs1[-1](x, edge)
        return node_score, x


class PredictionNet(torch.nn.Module):

    def __init__(self, in_features, num_class, hidden2):
        super(PredictionNet, self).__init__()

        self.hidden2 = hidden2
        self.convs2 = nn.ModuleList()
        self.batch_norms2 = nn.ModuleList()
        for i in range(len(self.hidden2)):
            if i == 0:
                conv = GraphConv_ordi(in_features=in_features, out_features=self.hidden2[0])
            else:
                conv = GraphConv_ordi(in_features=self.hidden2[i - 1], out_features=self.hidden2[i])
            self.convs2.append(conv)
            self.batch_norms2.append(nn.BatchNorm1d(self.hidden2[i]))

        self.causal_mlp = nn.Sequential(
            nn.Linear(sum(self.hidden2) * 2, self.hidden2[-1] * 4),
            # nn.BatchNorm1d(self.hidden2[-1]),
            nn.ReLU(True),
            nn.Linear(self.hidden2[-1] * 4, num_class)
        )
        self.conf_mlp = nn.Sequential(
            nn.Linear(sum(self.hidden2) * 2, self.hidden2[-1] * 4),
            # nn.BatchNorm1d(self.hidden2[-1]),
            nn.ReLU(True),
            nn.Linear(self.hidden2[-1] * 4, num_class)
        )

    def forward(self, x, edge):
        graph_x = self.get_graph_rep(x, edge)
        return self.get_causal_pred(graph_x)

    def get_graph_rep(self, x, edge):
        xs = []
        for conv, bn in zip(self.convs2, self.batch_norms2):
            x = conv(x, edge)
            x = F.relu(bn(x.transpose(1, 2)).transpose(1, 2))
            xs.append(x)
        return torch.cat([self.readout(xx) for xx in xs], dim=-1)  #

    def get_causal_pred(self, causal_graph_x):
        pred = self.causal_mlp(causal_graph_x)  # [32, 10]
        return pred

    def get_conf_pred(self, conf_graph_x):
        pred = self.conf_mlp(conf_graph_x)  # [32, 10]
        return pred

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_mlp(causal_graph_x)  # y_c
        conf_pred = self.conf_mlp(conf_graph_x).detach()  # y_s
        return torch.sigmoid(conf_pred) * causal_pred  # \hat{y}

    def readout(self, x):
        return torch.concat([global_max_pool(x), global_mean_pool(x)], dim=-1)


def global_max_pool(x):  # x:[B,P,d]
    return F.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze(2)


def global_mean_pool(x):  # x:[B,P,d]
    return F.avg_pool1d(x.transpose(1, 2), x.shape[1]).squeeze(2)


def split_graph_score_softmax(x, A, node_score, ratio, norm=False, tau=0.1):
    pass


def get_sub_graph(x, A, group, norm='DAD', type_int=0):
    pass

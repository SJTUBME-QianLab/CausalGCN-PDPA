import torch
import torch.nn as nn
import torch.nn.functional as F
from .GCN_layer import GCN as GraphConv_ordi


class ConvGCN(nn.Module):
    def __init__(self, in_features, num_class, hidden2):
        super(ConvGCN, self).__init__()

        # classification
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

        self.mlp = nn.Sequential(
            nn.Linear(sum(self.hidden2) * 2, self.hidden2[-1] * 4),
            # nn.BatchNorm1d(self.hidden2[-1]),
            nn.ReLU(True),
            nn.Linear(self.hidden2[-1] * 4, num_class)
        )

    def forward(self, x, edge):
        xs = []
        for conv, bn in zip(self.convs2, self.batch_norms2):
            x = conv(x, edge)
            x = F.relu(bn(x.transpose(1, 2)).transpose(1, 2))
            xs.append(x)
        graph = torch.cat([self.readout(xx) for xx in xs], dim=-1)  # [B,P,d*xx]
        out = self.mlp(graph)
        return out

    def readout(self, x):
        return torch.concat([global_max_pool(x), global_mean_pool(x)], dim=-1)


def global_max_pool(x):  # x:[B,P,d]
    return F.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze(2)


def global_mean_pool(x):  # x:[B,P,d]
    return F.avg_pool1d(x.transpose(1, 2), x.shape[1]).squeeze(2)

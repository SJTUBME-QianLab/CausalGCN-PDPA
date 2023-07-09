import torch
import torch.nn.functional as F
from torch.nn import Parameter
from net.brainmsgpassing import MyMessagePassing
from torch_geometric.utils import add_remaining_self_loops, softmax

from torch_geometric.typing import (OptTensor)

from net.inits import uniform


class MyNNConv(MyMessagePassing):  # net.brainmsgpassing
    def __init__(self, in_channels, out_channels, nn, normalize=False, bias=True, **kwargs):
        # in_channels=indim=200, out_channels=dim1=32, nn=self.n1 Ra_Gconv
        super(MyNNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.nn = nn
        # self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_weight=None, pseudo=None, size=None):
        """x: [400, 200]=[B*N,d0], pseudo: [400, 200]=[B*N,d0]"""
        edge_weight = edge_weight.squeeze()  # [79600, 1]
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1, x.size(0))

        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)  # nn:[B*N,d0*d1], view: [B*N,d0,d1]
        if torch.is_tensor(x):  # x: [B*N,1,d0], weight: [B*N,d0,d1],
            x = torch.matmul(x.unsqueeze(1), weight).squeeze(1)  # Eq.(1)  matmul: [B*N,1,d1] squeeze: [B*N,d1]
        else:
            x = (None if x[0] is None else torch.matmul(x[0].unsqueeze(1), weight).squeeze(1),
                 None if x[1] is None else torch.matmul(x[1].unsqueeze(1), weight).squeeze(1))

        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)
        # net.brainmsgpassing

    def message(self, edge_index_i, size_i, x_j, edge_weight, ptr: OptTensor):
        edge_weight = softmax(edge_weight, edge_index_i, ptr, size_i)
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

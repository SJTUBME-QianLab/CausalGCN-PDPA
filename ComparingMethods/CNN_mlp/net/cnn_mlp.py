import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEmbedding(nn.Module):
    def __init__(self, patch_num, num_class, hidden2, pool, feature_depth, kernels=None):
        super(ConvEmbedding, self).__init__()

        if kernels is None:
            kernels = [1] + [2] * (len(feature_depth)-2)
        assert len(feature_depth) == len(kernels)
        self.feature_depth = feature_depth
        padding = (lambda x: int((x-1)/2))
        self.patch_emb = nn.Sequential()
        for i in range(len(feature_depth)):
            if i == 0:
                self.patch_emb.add_module('conv_{}'.format(i), nn.Conv3d(1, feature_depth[i], kernel_size=kernels[i], padding=padding(kernels[i])))
            else:
                self.patch_emb.add_module('conv_{}'.format(i), nn.Conv3d(feature_depth[i-1], feature_depth[i], kernel_size=kernels[i], padding=padding(kernels[i])))
            self.patch_emb.add_module('norm_{}'.format(i), nn.BatchNorm3d(feature_depth[i]))
            self.patch_emb.add_module('relu_{}'.format(i), nn.ReLU(True))
        self.patch_emb.add_module('avgpool', nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)))

        self.pool = pool
        if self.pool.lower() == 'avg' or self.pool.lower() == 'max':
            self.mlp = MultiLayer(feature_depth[-1], num_class, hidden2)
        elif self.pool.lower() == 'cat':
            self.mlp = MultiLayer(feature_depth[-1] * patch_num, num_class, hidden2)
        else:
            raise ValueError(self.pool)

    def forward(self, x):  # [B,P,w,w,w]
        x_emb = self.emb(x)  # [B,P,d]
        if self.pool.lower() == 'avg':
            pool = F.avg_pool1d(x_emb.transpose(1, 2), x_emb.shape[1]).squeeze(2)  # [B,d,P]->[B,d,1]->[B,d]
            out = self.mlp(pool)  # [B,P*d]
        elif self.pool.lower() == 'max':
            Maxpool = F.max_pool1d(x_emb.transpose(1, 2), x_emb.shape[1]).squeeze(2)  # [B,d,P]->[B,d,1]->[B,d]
            out = self.mlp(Maxpool)  # [B,P*d]
        elif self.pool.lower() == 'cat':
            out = self.mlp(x_emb.reshape(x_emb.shape[0], -1))  # [B,P*d]
        else:
            raise ValueError(self.pool)
        return out

    def emb(self, x):  # [8, 200, 5, 5, 5]
        x_np = x.reshape(-1, *x.shape[2:]).unsqueeze(1)  # [1600, 1, 5, 5, 5]
        x_conv3d = self.patch_emb(x_np)  # [1600, d, 1, 1, 1]
        x_emb = x_conv3d.reshape(*x.shape[:2], -1)  # [8, 200, d]
        return x_emb  # [B,P,d]


class MultiLayer(nn.Module):
    def __init__(self, dim, num_class, hidden2):
        super(MultiLayer, self).__init__()

        self.mlp = nn.Sequential()
        if hidden2 is None:
            self.mlp = nn.Linear(dim, num_class)
        else:
            for i in range(len(hidden2)):
                if i == 0:
                    self.mlp.add_module('lin_{}'.format(i), nn.Linear(dim, hidden2[i]))
                else:
                    self.mlp.add_module('lin_{}'.format(i), nn.Linear(hidden2[i-1], hidden2[i]))
                self.mlp.add_module('relu_{}'.format(i), nn.ReLU(True))
            self.mlp.add_module('lin_{}'.format(len(hidden2)), nn.Linear(hidden2[-1], num_class))

    def forward(self, x_emb):  # [B,P,d]
        out = self.mlp(x_emb)  # [B,P*d]
        return out


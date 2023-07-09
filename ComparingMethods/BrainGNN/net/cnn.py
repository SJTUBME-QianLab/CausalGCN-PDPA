import torch
import torch.nn as nn


class ConvEmbedding(nn.Module):
    def __init__(self, feature_depth, kernels=None):
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

    def forward(self, x):  # [8, 200, 5, 5, 5]
        x_np = x.reshape(-1, *x.shape[2:]).unsqueeze(1)  # [1600, 1, 5, 5, 5]
        x_conv3d = self.patch_emb(x_np)  # [1600, d, 1, 1, 1]
        x_emb = x_conv3d.reshape(*x.shape[:2], -1)  # [8, 200, d]
        return x_emb  # [N,P,d]

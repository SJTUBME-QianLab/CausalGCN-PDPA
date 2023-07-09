"""
reference: https://github.com/WyZhuNUAA/DA-MIDL/
"""
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.cs = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):  # [B,128,w,w,w]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,w,w,w] [8, 1, 6, 6, 6]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,w,w,w]
        a = torch.cat([avg_out, max_out], dim=1)  # [B,2,w,w,w]
        a = self.cs(a)  # [B,1,w,w,w]
        return x*a  # [B,128,w,w,w]


class AttentionBlock(nn.Module):
    def __init__(self, patch_num):
        super(AttentionBlock, self).__init__()
        self.patch_num = patch_num
        self.GlobalAveragePool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))  # GAP Eq.5
        self.GlobalMaxPool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))  # GMP Eq.6
        self.Attn = nn.Sequential(
            nn.Conv3d(self.patch_num, self.patch_num // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.patch_num // 2, self.patch_num, kernel_size=1)
        )
        self.pearson_attn = nn.Linear(self.patch_num - 1, 1)

    def forward(self, input, patch_pred):  # [8, 10, 128, 6, 6, 6], [8, 10]
        mean_input = input.mean(2)  # [8, 10, 6, 6, 6]
        attn1 = self.Attn(self.GlobalAveragePool(mean_input))  # GAP Eq.5 [8, 10, 1, 1, 1] Attn: [8, 10, 1, 1, 1]
        attn2 = self.Attn(self.GlobalMaxPool(mean_input))  # GMP Eq.6
        patch_pred = patch_pred.unsqueeze(-1)
        patch_pred = patch_pred.unsqueeze(-1)
        patch_pred = patch_pred.unsqueeze(-1)  # [8, 10, 1, 1, 1]
        a = attn1 + attn2 + patch_pred  # Eq.7
        a = torch.sigmoid(a)  #
        return mean_input*a, a.flatten(1)  # [8, 10, 6, 6, 6], [8,10]


class BaseNet(nn.Module):
    def __init__(self, feature_depth):
        super(BaseNet, self).__init__()
        self.feature_depth = feature_depth
        self.spatial_attention = SpatialAttention()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, self.feature_depth[0], kernel_size=2)),
            ('norm1', nn.BatchNorm3d(self.feature_depth[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(self.feature_depth[0], self.feature_depth[1], kernel_size=2)),
            ('norm2', nn.BatchNorm3d(self.feature_depth[1])),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool3d(kernel_size=1)),
            ('conv3', nn.Conv3d(self.feature_depth[1], self.feature_depth[2], kernel_size=1)),
            ('norm3', nn.BatchNorm3d(self.feature_depth[2])),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv3d(self.feature_depth[2], self.feature_depth[3], kernel_size=1)),
            ('norm4', nn.BatchNorm3d(self.feature_depth[3])),
            ('relu4', nn.ReLU(inplace=True)),
        ]))
        self.classify = nn.Sequential(
            nn.Linear(self.feature_depth[3], 1),
            nn.Sigmoid()
        )

    def forward(self, input):  # [8, 1, 25, 25, 25]
        local_feature = self.features(input)  # [8, 128, 6, 6, 6]  (in+2*padding-kernel)/stride+1
        attended_feature = self.spatial_attention(local_feature)  # [8, 128, 6, 6, 6]
        feature_ = F.adaptive_avg_pool3d(local_feature, (1, 1, 1))  # [8, 128, 1, 1, 1]
        score = self.classify(feature_.flatten(1, -1))  # [8, 1]
        return [attended_feature, score]


class DAMIDL(nn.Module):
    def __init__(self, patch_num=60, feature_depth=None):
        super(DAMIDL, self).__init__()
        self.patch_num = patch_num
        if feature_depth is None:
            feature_depth = [32, 64, 128, 128]
        self.patch_net = BaseNet(feature_depth)
        self.attention_net = AttentionBlock(self.patch_num)
        self.reduce_channels = nn.Sequential(
            nn.Conv3d(self.patch_num, 128, kernel_size=2),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 64, kernel_size=2),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, input):  # [B,P,w,w,w]
        input = input.transpose(0, 1).unsqueeze(2)
        # input: [num_patches, batch_size, 1, patch_size, patch_size, patch_size] = [P,B,1,w,w,w]
        patch_feature, patch_score = [], []
        for i in range(self.patch_num):  # [B, 1, w, w, w]
            feature, score = self.patch_net(input[i])  # [8, 128, 6, 6, 6], [8, 1]
            feature = feature.unsqueeze(1)  # [8, 1, 128, 6, 6, 6]
            patch_feature.append(feature)
            patch_score.append(score)
        feature_maps = torch.cat(patch_feature, 1)  # [8, 10, 128, 6, 6, 6] = [B,P,d,w,w,w]
        patch_scores = torch.cat(patch_score, 1)  # [8, 10] = [B,P]
        attn_feat, ca = self.attention_net(feature_maps, patch_scores)  # [8, 10, 6, 6, 6], [8,10]
        features = self.reduce_channels(attn_feat).flatten(1)  # [8, 64, 1, 1, 1], [8, 64]
        subject_pred = self.fc(features)  # [8,2]
        return subject_pred, ca


# import numpy as np
# input_shape = (8, 1, 25, 25, 25)
# num_patches = 10
# inputs = []
# for i_input in range(num_patches):
#     inputs.append(np.zeros(input_shape, dtype='float32'))
# inputs = torch.from_numpy(np.array(inputs))  # [10, 8, 1, 25, 25, 25] = [P, B, 1, w, w, w]
# # [num_patches, batch_size, 1, patch_size, patch_size, patch_size]
# Net = DAMIDL(patch_num=num_patches)
# out = Net(inputs)

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.transformer.pos_embeding import *
from models.transformer.NetVLAD import *
from models.transformer.dgcnn import get_graph_feature_per_sector

import matplotlib.pyplot as plt

def visual_pc(x):
    for j in range(x.shape[0]):
        # 三维点可视化
        pc_tT = x[j][:, 0:3].numpy()
        x_rawT = pc_tT[:, 0]
        y_rawT = pc_tT[:, 1]
        z_rawT = pc_tT[:, 2]

        fig = plt.figure()
        ax_raw = fig.add_subplot(111, projection='3d')

        color_raw = np.arange(pc_tT.shape[0])

        ax_raw.scatter(x_rawT, y_rawT, z_rawT, c=color_raw, cmap='viridis', s=1)

        ax_raw.set_xlim(-100, 100)
        ax_raw.set_ylim(-100, 100)
        ax_raw.set_zlim(-100, 100)

        ax_raw.set_title('Raw Visualization')

        plt.tight_layout()
        plt.show()


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PNT_GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, kernel=(4096,1)):
        super(PNT_GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.kernel = kernel

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), kernel_size=self.kernel).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class PPLoc3D(nn.Module):
    def __init__(self, combine_params=None, num_points=4096, output_dim=256):
        super(PPLoc3D, self).__init__()

        self.t_net = combine_params['pplnet']['t_net']
        self.sectors = combine_params['pplnet']['sectors']
        self.npoint = combine_params['pplnet']['npoint']
        self.k = combine_params['pplnet']['knn']
        self.d = combine_params['pplnet']['divide']
        self.p_num_reduce = combine_params['pplnet']['p_num_reduce']
        self.num_points = num_points
        self.output_dim = output_dim

        assert self.num_points == self.sectors*self.npoint, 'num_points must be equal sectors*npoint'

        self.conv0 = torch.nn.Conv2d(48, 64, (1, 1))
        self.bn0 = nn.BatchNorm2d(64)

        self.conv1 = torch.nn.Conv2d(64, 128, (1, 1))
        self.bn1 = nn.BatchNorm2d(128)

        self.convDG1 = torch.nn.Conv2d(2*64, 2*64, (1, 1))
        self.bnDG1 = nn.BatchNorm2d(2*64)

        if self.t_net:
            self.stn = STN3d(num_points=self.num_points, k=3, use_bn=False)
            # self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)

            self.pos_ecoding_l = PositionalEncodingLocal(d_features_in=16, d_features_out=64,
                                                         sectors=self.sectors // self.d, npoint=self.npoint * self.d)

            self.conv2_add = torch.nn.Conv2d(128, 32, (1, 1))
            self.bn2_add = nn.BatchNorm2d(32)
        else:
            self.npoint = int(self.npoint/self.p_num_reduce)

            self.pos_ecoding_l = PositionalEncodingLocal(d_features_in=16, d_features_out=64,
                                                         sectors=self.sectors // self.d, npoint=self.npoint * self.d)

            self.conv2 = torch.nn.Conv2d(128, 128, (1, 1))
            self.bn2 = nn.BatchNorm2d(128)
            self.conv2_add = torch.nn.Conv2d(256, 32, (1, 1))
            self.bn2_add = nn.BatchNorm2d(32)
            self.conv3 = torch.nn.Conv2d(256, 256, (1, 1))
            self.bn3 = nn.BatchNorm2d(256)

            self.mp1 = torch.nn.MaxPool2d((1, self.npoint), 1)

            self.pos_ecoding_g = PositionalEncodingGlobal(sectors=self.sectors)

            encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=1, dim_feedforward=256, activation='relu')
            self.att2 = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

            self.pntnet_pooling = PNT_GeM(kernel=(self.num_points, 1))


    def forward(self, x):
        batch_size = x.shape[0]

        # visual_pc(x[0].squeeze(0).cpu())

        if self.t_net:
            trans = self.stn(x)
            x = torch.matmul(torch.squeeze(x), trans)
            x = x.view(batch_size, 1, -1, 3)

        if self.t_net:
            x = x.view(batch_size, self.sectors, self.npoint, 3)
            x_r = torch.norm(x[:, :, :, 0:2], p=2, dim=-1).unsqueeze(-1)
            x_z = x[:, :, :, 2:]
            x = torch.cat((x_r, x_z), dim=-1)
        else:
            x = x.view(batch_size, self.sectors, self.npoint*self.p_num_reduce, 3)
            x = x[:, :, :self.npoint, :]
            plnt_x = x.contiguous().view(batch_size, -1, 3)
            # visual_pc(x.contiguous().view(batch_size, -1, 3).cpu())
            x_r = torch.norm(x[:, :, :, 0:2], p=2, dim=-1).unsqueeze(-1)
            x_z = x[:, :, :, 2:]
            x = torch.cat((x_r, x_z), dim=-1)
            _, indics = torch.topk(x_r, self.npoint, dim=2, largest=True)
            x = torch.gather(x, 2, indics.repeat(1, 1, 1, 2))
            x = torch.flip(x, dims=[2])

        x = x.permute(0, 3, 1, 2)

        x = x.view(batch_size, -1, self.sectors//self.d, self.npoint*self.d)
        x = self.pos_ecoding_l(x)
        x = x.view(batch_size, -1, self.sectors, self.npoint)
        x = F.relu(self.bn0(self.conv0(x)))

        # if self.t_net:
        #     x = x.view(batch_size, -1, self.sectors*self.npoint, 1)
        #     f_trans = self.feature_trans(x)
        #     x = torch.squeeze(x)
        #     if batch_size == 1:
        #         x = torch.unsqueeze(x, 0)
        #     x = torch.matmul(x.transpose(1, 2), f_trans)
        #     x = x.transpose(1, 2).contiguous()
        #     x = x.view(batch_size, 64, -1, 1)

        x_t = x.permute(0, 2, 1, 3).contiguous().view(batch_size * self.sectors//self.d, -1, self.npoint*self.d)
        x_t = get_graph_feature_per_sector(x_t, k=self.k, cat=True)
        x_t = x_t.contiguous().view(batch_size, self.sectors//self.d, -1, self.npoint*self.d, self.k)
        x_t = x_t.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, -1, self.sectors*self.npoint, self.k)
        x_t = F.relu(self.bnDG1(self.convDG1(x_t)))
        x_t = x_t.max(dim=-1, keepdim=True)[0]
        x = x_t.contiguous().view(batch_size, -1, self.sectors, self.npoint)

        if self.t_net:
            x = F.relu(self.bn2_add(self.conv2_add(x)))
            x = x.view(batch_size, -1, self.sectors * self.npoint, 1)
            return x

        x = F.relu(self.bn2(self.conv2(x)))
        x_feat = self.mp1(x)
        x_feat = x_feat.permute(0, 1, 3, 2)
        x_feat = self.pos_ecoding_g(x_feat).squeeze(2).permute(2, 0, 1)
        x_feat = self.att2(x_feat)
        x_feat = x_feat.permute(1, 2, 0).unsqueeze(3).repeat(1, 1, 1, self.npoint)
        x_feat = torch.cat((x, x_feat), dim=1).view(batch_size, -1, self.sectors*self.npoint, 1)

        x = F.relu(self.bn2_add(self.conv2_add(x_feat)))

        return plnt_x, x, x_feat
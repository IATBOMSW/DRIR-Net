import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math

class PositionalEncodingLocal(nn.Module):
    def __init__(self, d_features_in=16, d_features_out=64, sectors=256, npoint=16):
        super(PositionalEncodingLocal, self).__init__()

        self.d_features_in = d_features_in
        self.d_features_out = d_features_out
        self.sectors = sectors
        self.npoint = npoint

        alpha = 2 * math.pi / sectors
        beta = npoint
        pe = torch.zeros(d_features_in * 2)

        div_term = torch.exp(torch.arange(0, d_features_in * 2, 2).float() / d_features_in * 2 * (-math.log(d_features_in)))

        pe[0::2] = torch.cos(alpha * div_term)
        pe[1::2] = torch.sin(alpha * div_term)


        self.register_buffer('pe', pe)

        # self.w = nn.Parameter(torch.randn(d_features_in * 3, d_features_out) * 1 / math.sqrt(2*d_features_out/3))


    def forward(self, x):
        """
        x: [B, 2, S, npoint]

        x_emb: [B, 3*L, S, npoint]
        """
        x_pe = torch.zeros([x.shape[0], 3*self.d_features_in, self.sectors, self.npoint], dtype=torch.float).to(x.device)

        x_r = x[:, 0, :, :].unsqueeze(1)
        # x_r = x_r * self.pe.view(1, -1, 1, 1)

        x_pe[:, 0::3, :, :] = x_r * (self.pe[0::2]).view(1, -1, 1, 1)
        x_pe[:, 1::3, :, :] = x_r * (self.pe[1::2]).view(1, -1, 1, 1)

        x_z = x[:, 1, :, :].unsqueeze(1).repeat(1, self.d_features_in, 1, 1)

        x_pe[:, 2::3, :, :] = x_z

        # x = torch.cat((x_r, x_z), dim=1)

        # x = torch.matmul(x.permute(0, 2, 3, 1), self.w).permute(0, 3, 1, 2)

        return x_pe


class PositionalEncodingGlobal(nn.Module):
    def __init__(self, sectors=64):
        super(PositionalEncodingGlobal, self).__init__()

        pe = torch.zeros(sectors)
        position = torch.arange(0, sectors, dtype=torch.float)
        div_term = math.pi / sectors
        pe[0::2] = torch.sin(position * div_term)[0::2]
        pe[1::2] = torch.cos(position * div_term)[1::2]

        self.register_buffer('pe', pe)

        self.weights = nn.Parameter(torch.randn(sectors, sectors) * 1 / math.sqrt(sectors))

    def forward(self, x):
        """
        x: [B, C, N, S]

        x_emb: [B, C+p, N, S]
        """

        pe = torch.matmul(self.pe.unsqueeze(0), self.weights)

        x = x + pe.view(1, 1, 1, -1).repeat(1, 1, x.shape[2], 1)

        return x
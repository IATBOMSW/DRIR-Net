import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

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

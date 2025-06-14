# Author: Jacek Komorowski
# Warsaw University of Technology
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from datasets.dataset_utils import to_spherical
from models.minkfpn import MinkFPN
from models.netvlad import MinkNetVladWrapper
import layers.pooling as layers_pooling
from typing import List
from models.transformer.pipline import PPLoc3D
from models.transformer.NetVLAD import Gate
from models.pointnet.PointNet import PointNetfeatv1


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



class MinkLoc(torch.nn.Module):
    def __init__(self, model, pooling, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size,
                 block_module, num_points, dataset_name, combine_params):
        super().__init__()
        self.model = model
        self.with_pointnet = True if 'pointnet' in combine_params else False
        self.with_pplnet = True if 'pplnet' in combine_params else False
        self.with_cross_att = True if 'cross_attention' in combine_params else False
        self.t_net = combine_params['pplnet']['t_net']
        self.planes = planes

        self.in_channels = in_channels
        self.feature_size = feature_size    # Size of local features produced by local feature extraction block
        self.output_dim = output_dim        # Dimensionality of the global descriptor
        self.backbone = MinkFPN(in_channels=in_channels, out_channels=self.feature_size, num_top_down=num_top_down,
                                conv0_kernel_size=conv0_kernel_size, block=block_module, layers=layers, planes=planes, combine_params=combine_params)
        self.n_backbone_features = output_dim

        self.dataset_name = dataset_name

        if pooling == 'Max':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.pooling = layers_pooling.MAC()
        elif pooling == 'GeM':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.pooling = layers_pooling.GeM()
        elif pooling == 'NetVlad':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, output_dim=self.output_dim,
                                              cluster_size=64, gating=False)
        elif pooling == 'NetVlad_CG':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, output_dim=self.output_dim,
                                              cluster_size=64, gating=True)
        else:
            raise NotImplementedError('Model not implemented: {}'.format(model))

        if self.with_pointnet:
            self.pointnet = PointNetfeatv1(num_points=num_points,
                                           global_feat=True,
                                           feature_transform=True,
                                           max_pool=False,
                                           output_dim=feature_size if self.with_pointnet else planes[0])

        if self.with_pplnet:
            self.p_num_reduce = combine_params['pplnet']['p_num_reduce']
            self.fusion = combine_params['pplnet']['fusion']
            self.pw_rat = combine_params['pplnet']['pw_rat']
            self.vw_rat = combine_params['pplnet']['vw_rat']
            self.ppl = PPLoc3D(combine_params=combine_params, num_points=num_points, output_dim=self.output_dim)

            if self.fusion:
                self.conv = torch.nn.Conv2d(32, 256, (1, 1))
                self.bn = nn.BatchNorm2d(256)
                self.pntnet_pooling = PNT_GeM(kernel=(int(num_points/self.p_num_reduce), 1))


    def forward(self, batch):
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        feats = batch['features']
        feats = feats.to('cuda')
        coords = batch['coords']
        coords = coords.to('cuda')

        x = ME.SparseTensor(feats, coords)

        if self.with_pplnet:
            PLNT_x = batch['plnt_coords']
            if self.t_net:
                PLNT_feats = self.ppl(PLNT_x.unsqueeze(dim=1))
            else:
                PLNT_x, PLNT_feats, PLNT_t = self.ppl(PLNT_x.unsqueeze(dim=1))

        if self.with_cross_att:
            PLNT_x_list = [item for item in PLNT_x]
            PLNT_coords = ME.utils.batched_coordinates(PLNT_x_list).to(PLNT_x.device)
            assert type(self.backbone).__name__ == 'MinkFPN', 'backbone for cross attention should be MinkFPN'
            x, y = self.backbone(x, PLNT_coords, PLNT_feats.squeeze(dim=-1).view(-1, self.planes[0]))
        else:
            x, _ = self.backbone(x)

        if self.with_pplnet and self.fusion:
            y = y.permute(1, 2, 0).unsqueeze(-1)
            y = F.relu(self.bn(self.conv(y)))
            #y = y + PLNT_t
            y = self.pntnet_pooling(y).view(-1, self.output_dim)

        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.feature_size)
        x = self.pooling(x)

        assert x.dim() == 2, 'Expected 2-dimensional tensor (batch_size,output_dim). Got {} dimensions.'.format(x.dim())
        assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.output_dim)
        # x is (batch_size, output_dim) tensor

        if self.with_pplnet and self.fusion:
            x = self.vw_rat * x + self.pw_rat * y

        return x

    def print_info(self):
        print('Model class: MinkLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print('Backbone parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print('Aggregation parameters: {}'.format(n_params))
        if hasattr(self.backbone, 'print_info'):
            self.backbone.print_info()
        if hasattr(self.pooling, 'print_info'):
            self.pooling.print_info()
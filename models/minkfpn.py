# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from models.resnet import ResNetBase
from typing import List
from torchtyping import TensorType
from models.transformer.transformers import TransformerCrossEncoderLayer, TransformerCrossEncoder
from models.transformer.position_embedding import PositionEmbeddingLearned

def pad_sequence(sequences, require_padding_mask=False, require_lens=False,
                 batch_first=False):
    """List of sequences to padded sequences

    Args:
        sequences: List of sequences (N, D)
        require_padding_mask:

    Returns:
        (padded_sequence, padding_mask), where
           padded sequence has shape (N_max, B, D)
           padding_mask will be none if require_padding_mask is False
    """
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    padding_mask = None
    padding_lens = None

    if require_padding_mask:
        B = len(sequences)
        seq_lens = list(map(len, sequences))
        padding_mask = torch.zeros((B, padded.shape[0]), dtype=torch.bool, device=padded.device)
        for i, l in enumerate(seq_lens):
            padding_mask[i, l:] = True

    if require_lens:
        padding_lens = [seq.shape[0] for seq in sequences]

    return padded, padding_mask, padding_lens


def unpad_sequences(padded, seq_lens):
    """Reverse of pad_sequence"""
    sequences = [padded[..., :seq_lens[b], b, :] for b in range(len(seq_lens))]
    return sequences

class MinkFPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(self, in_channels, out_channels, num_top_down=1, conv0_kernel_size=5, block=BasicBlock,
                 layers=(1, 1, 1), planes=(32, 64, 64), combine_params=None):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)

        self.with_cross_att = True if 'cross_attention' in combine_params else False
        if self.with_cross_att:
            cross_att_key = 'cross_attention' if 'cross_attention' in combine_params else 'multi_cross_attention'

        if self.with_cross_att:
            d_embed = planes[0] # cross attention after first layer of conv
            self.transformer_encoder_has_pos_emb = combine_params[cross_att_key]['transformer_encoder_has_pos_emb']
            self.pos_embed = PositionEmbeddingLearned(3, d_embed)

            encoder_layer = TransformerCrossEncoderLayer(
                d_model=d_embed,
                nhead=combine_params[cross_att_key]['nhead'],
                dim_feedforward=combine_params[cross_att_key]['d_feedforward'],
                dropout=combine_params[cross_att_key]['dropout'],
                activation=combine_params[cross_att_key]['transformer_act'],
                normalize_before=combine_params[cross_att_key]['pre_norm'],
                sa_val_has_pos_emb=combine_params[cross_att_key]['sa_val_has_pos_emb'],
                ca_val_has_pos_emb=combine_params[cross_att_key]['ca_val_has_pos_emb'],
                attention_type=combine_params[cross_att_key]['attention_type'],
            )
            self.transformer_encoder = TransformerCrossEncoder(
                cross_encoder_layer=encoder_layer,
                num_layers=combine_params[cross_att_key]['num_encoder_layers'],
                norm=nn.LayerNorm(d_embed) if combine_params[cross_att_key]['pre_norm'] else None,
                return_intermediate=False)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()    # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()       # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()   # Bottom-up blocks
        self.tconvs = nn.ModuleList()   # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size,
                                             dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)


        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - i], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
            self.tconvs.append(ME.MinkowskiConvolutionTranspose(self.lateral_dim, self.lateral_dim, kernel_size=2,
                                                                stride=2, dimension=D))
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - self.num_top_down], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[0], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))

        self.relu = ME.MinkowskiReLU(inplace=True)



    def batch_feat_size(self,x) -> List[int]:
        _, batch_feat_size = torch.unique(x[:,0], return_counts=True)
        return batch_feat_size.tolist()

    def batch_tolist(self, x:TensorType, seq:List[int]) -> List[TensorType]:
        x = list(torch.split(x, seq))
        return x


    def combine_cross_attention(self, x, y_c, y_f, time_file=None):
        x_batch_feat_size = self.batch_feat_size(x.C)
        y_batch_feat_size = self.batch_feat_size(y_c)

        x_pe = self.batch_tolist(self.pos_embed(x.C[:, 1:].to(torch.float)), x_batch_feat_size)
        y_pe = self.batch_tolist(self.pos_embed(y_c[:, 1:].to(torch.float)), y_batch_feat_size)
        y_feats_un = self.batch_tolist(y_f, y_batch_feat_size)
        x_feats_un = self.batch_tolist(x.F, x_batch_feat_size)

        x_pe_padded, _, _ = pad_sequence(x_pe)
        y_pe_padded, _, _ = pad_sequence(y_pe)

        x_feats_padded, x_key_padding_mask, _ = pad_sequence(x_feats_un,
                                                                require_padding_mask=True)
        y_feats_padded, y_key_padding_mask, _ = pad_sequence(y_feats_un,
                                                                require_padding_mask=True)

        x_feats_cond, y_feats_cond, cross_attention_time_dict = self.transformer_encoder(
            x_feats_padded, y_feats_padded,
            src_key_padding_mask=x_key_padding_mask,
            tgt_key_padding_mask=y_key_padding_mask,
            src_pos=x_pe_padded if self.transformer_encoder_has_pos_emb else None,
            tgt_pos=y_pe_padded if self.transformer_encoder_has_pos_emb else None,
            time_file=time_file
        )

        x_feats_cond = torch.squeeze(x_feats_cond, dim=0)
        y_feats_cond = torch.squeeze(y_feats_cond, dim=0)
        x_feats_list = unpad_sequences(x_feats_cond, x_batch_feat_size)
        # y_feats_list = unpad_sequences(y_feats_cond, y_batch_feat_size)

        x_feats = torch.vstack(x_feats_list)
        # y_feats = torch.vstack(y_feats_list)

        x_feats = x.F + x_feats

        x = ME.SparseTensor(coordinates=x.C, features=x_feats)

        return x, y_feats_cond, cross_attention_time_dict

    def forward(self, x, y_c=None, y_f=None):
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger stride)
        feature_maps = []

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        y_f_att = None
        if self.with_cross_att:
            x_a, y_f_att, cross_attention_time_dict = self.combine_cross_attention(x, y_c, y_f)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)     # Decreases spatial resolution (conv stride=2)
            x = bn(x)
            x = self.relu(x)
            x = block(x)
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1[0](x)

        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)        # Upsample using transposed convolution
            x = x + self.conv1x1[ndx+1](feature_maps[-ndx - 1])

        return x, y_f_att
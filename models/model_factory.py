# Author: Jacek Komorowski
# Warsaw University of Technology

import models.minkloc as minkloc
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from layers.eca_block import ECABasicBlock
import torch.nn as nn

def model_factory(params):
    in_channels = 1

    if 'MinkFPN' in params.model_params.model:
        block_module = create_resnet_block(params.model_params.block)
        model = minkloc.MinkLoc(params.model_params.model,
                                pooling=params.model_params.pooling,
                                in_channels=in_channels,
                                feature_size=params.model_params.feature_size,
                                output_dim=params.model_params.output_dim, planes=params.model_params.planes,
                                layers=params.model_params.layers, num_top_down=params.model_params.num_top_down,
                                conv0_kernel_size=params.model_params.conv0_kernel_size,
                                block_module=block_module,
                                num_points=params.num_points,
                                dataset_name=params.dataset_name,
                                combine_params=params.model_params.combine_params)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(params.model_params.model))

    return model

def create_resnet_block(block_name: str) -> nn.Module:
    if block_name == 'BasicBlock':
        block_module = BasicBlock
    elif block_name == 'Bottleneck':
        block_module = Bottleneck
    elif block_name == 'ECABasicBlock':
        block_module = ECABasicBlock
    else:
        raise NotImplementedError('Unsupported network block: {}'.format(block_name))

    return block_module
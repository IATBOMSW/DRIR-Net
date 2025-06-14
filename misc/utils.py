# Author: Jacek Komorowski
# Warsaw University of Technology
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)


import os
import configparser
import time
import numpy as np
from typing import Dict

class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.gpu = params.getint('gpu')
        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.output_dim = params.getint('output_dim', 256)  # Size of the final descriptor

        # Add gating as the last step
        if 'vlad' in self.model.lower():
            self.cluster_size = params.getint('cluster_size', 64)  # Size of NetVLAD cluster
            self.gating = params.getboolean('gating', True)  # Use gating after the NetVlad

        #######################################################################
        # Model dependent
        #######################################################################

        if 'MinkFPN' in self.model:
            # Models using MinkowskiEngine
            self.mink_quantization_size = [float(item) for item in params['mink_quantization_size'].split(',')]
            self.version = params['version']
            assert self.version in ['MinkLoc3D', 'MinkLoc3D-I', 'MinkLoc3D-S', 'MinkLoc3D-SI'], 'Supported versions ' \
                                                                                                'are: MinkLoc3D, ' \
                                                                                                'MinkLoc3D-I, ' \
                                                                                                'MinkLoc3D-S, ' \
                                                                                                'MinkLoc3D-SI '
            # Size of the local features from backbone network (only for MinkNet based models)
            # For PointNet-based models we always use 1024 intermediary features
            self.feature_size = params.getint('feature_size', 256)
            if 'planes' in params:
                self.planes = [int(e) for e in params['planes'].split(',')]
            else:
                self.planes = [32, 64, 64]

            if 'layers' in params:
                self.layers = [int(e) for e in params['layers'].split(',')]
            else:
                self.layers = [1, 1, 1]

            self.num_top_down = params.getint('num_top_down', 1)
            self.conv0_kernel_size = params.getint('conv0_kernel_size', 5)
            self.block = params.get('block', 'BasicBlock')

            self.pooling = params.get('pooling')
            assert self.pooling in ['Max', 'GeM', 'NetVlad',
                                    'NetVlad_CG'], 'Supported Pooling are: Max, GeM, NetVlad, NetVlad_CG'
            combine_modules = ['POINTNET', 'PPLNET', 'CROSS-ATTENTION'] \
                              if self.model == 'MinkFPN' else None
            combine_modules = {} if self.version not in ['MinkLoc3D-S', 'MinkLoc3D-SI', 'MinkLoc3D'] else combine_modules

            self.get_combine_params(config, combine_modules)
            assert isinstance(self.combine_params, Dict)

    def get_combine_params(self, config, combine_modules):
        self.combine_params = {}
        pointnet_params = config['POINTNET'] if 'POINTNET' in combine_modules else None
        pplnet_params = config['PPLNET'] if 'PPLNET' in combine_modules else None
        cross_att_params = config['CROSS-ATTENTION'] if 'CROSS-ATTENTION' in combine_modules else None

        with_pointnet = pointnet_params.getboolean('with_pointnet') if pointnet_params is not None else None
        with_pplnet = pplnet_params.getboolean('with_pplnet') if pplnet_params is not None else None
        with_cross_att = cross_att_params.getboolean('with_cross_att') if cross_att_params is not None else None

        if with_pointnet:
            pntnet_combine_params = {'pointnet':
                                         { 'pnts': pointnet_params.getboolean('pnts') }}
            self.combine_params = {**self.combine_params, **pntnet_combine_params}

        if with_pplnet:
            pplnet_combine_params = {'pplnet':
                                         { 't_net': pplnet_params.getboolean('t_net'),
                                            'sectors': pplnet_params.getint('sectors'),
                                           'npoint': pplnet_params.getint('npoint'),
                                           'knn': pplnet_params.getint('knn'),
                                           'divide': pplnet_params.getint('divide'),
                                           'p_num_reduce': pplnet_params.getint('p_num_reduce'),
                                           'fusion': pplnet_params.getboolean('fusion'),
                                           'pw_rat': pplnet_params.getfloat('pw_rat'),
                                           'vw_rat': pplnet_params.getfloat('vw_rat')}}
            self.combine_params = {**self.combine_params, **pplnet_combine_params}


        if with_cross_att:
            assert cross_att_params['attention_type'] in ['dot_prod', 'linear_attention'], 'Supported attention types: dot_prod, linear_attention'
            cross_att_combine_params = {"cross_attention":
                                                           {"nhead": cross_att_params.getint('num_heads'),
                                                            "d_feedforward": cross_att_params.getint("d_feedforward"),
                                                            "dropout": cross_att_params.getint("dropout"),
                                                            "transformer_act": cross_att_params['transformer_act'],
                                                            "pre_norm": cross_att_params.getboolean("pre_norm"),
                                                            "attention_type": cross_att_params['attention_type'],
                                                            "sa_val_has_pos_emb": cross_att_params.getboolean('sa_val_has_pos_emb'),
                                                            "ca_val_has_pos_emb": cross_att_params.getboolean('ca_val_has_pos_emb'),
                                                            "num_encoder_layers": cross_att_params.getint('num_encoder_layers'),
                                                            "transformer_encoder_has_pos_emb": cross_att_params.getboolean('transformer_encoder_has_pos_emb') }}
            self.combine_params = {**self.combine_params, **cross_att_combine_params}


    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))

        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


def xyz_from_depth(depth_image, depth_intrinsic, depth_scale=1000.):
    # Return X, Y, Z coordinates from a depth map.
    # This mimics OpenCV cv2.rgbd.depthTo3d() function
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    # Construct (y, x) array with pixel coordinates
    y, x = np.meshgrid(range(depth_image.shape[0]), range(depth_image.shape[1]), sparse=False, indexing='ij')

    X = (x - cx) * depth_image / (fx * depth_scale)
    Y = (y - cy) * depth_image / (fy * depth_scale)
    xyz = np.stack([X, Y, depth_image / depth_scale], axis=2)
    xyz[depth_image == 0] = np.nan
    return xyz


class MinkLocParams:
    """
    Params for training MinkLoc models on Oxford dataset
    """

    def __init__(self, params_path, model_params_path):
        """
        Configuration files
        :param path: General configuration file
        :param model_params: Model-specific configuration
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(
            model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.model_params_path = model_params_path

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.num_points = params.getint('num_points')
        self.max_distance = params.getint('max_distance')

        self.dataset_name = params.get('dataset_name')
        assert self.dataset_name in ['USyd', 'IntensityOxford', 'Oxford', 'kitti'], 'Dataset should be USyd, IntensityOxford, kitti ' \
                                                                           'or Oxford '

        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.num_workers = params.getint('num_workers', 0)
        self.batch_size = params.getint('batch_size', 128)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.lr = params.getfloat('lr', 1e-3)

        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                scheduler_milestones = params.get('scheduler_milestones')
                self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.epochs = params.getint('epochs', 20)
        self.weight_decay = params.getfloat('weight_decay', None)
        self.normalize_embeddings = params.getboolean('normalize_embeddings',
                                                      True)  # Normalize embeddings during training and evaluation
        self.loss = params.get('loss')

        if 'Contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'Triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)  # Margin used in loss function
        elif self.loss == 'truncatedsmoothap':
            # Number of best positives (closest to the query) to consider
            self.positives_per_query = params.getint("positives_per_query", 4)
            # Temperatures (annealing parameter) and numbers of nearest neighbours to consider
            self.tau1 = params.getfloat('tau1', 0.01)
            self.margin = params.getfloat('margin', None)    # Margin used in loss function
            # Similarity measure: based on cosine similarity or Euclidean distance
            self.similarity = params.get('similarity', 'euclidean')
            assert self.similarity in ['cosine', 'euclidean']
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.aug_mode = params.getint('aug_mode', 1)  # Augmentation mode (1 is default)

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.resume = params.getint('resume', 0)
        self.resume_name = params.get('resume_name')

        if self.dataset_name == 'USyd':
            self.eval_database_files = ['USyd_evaluation_database.pickle']
            self.eval_query_files = ['USyd_evaluation_query.pickle']

        elif self.dataset_name == 'IntensityOxford':
            self.eval_database_files = ['IntensityOxford_evaluation_database.pickle']
            self.eval_query_files = ['IntensityOxford_evaluation_query.pickle']

        elif self.dataset_name == 'Oxford':
            self.eval_database_files = ['oxford_evaluation_database.pickle']
            self.eval_query_files = ['oxford_evaluation_query.pickle']

        elif self.dataset_name == 'kitti':
            self.eval_database_files = ['kitti_evaluation_database.pickle']
            self.eval_query_files = ['kitti_evaluation_query.pickle']

        assert len(self.eval_database_files) == len(self.eval_query_files)

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)

        self._check_params()


    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')

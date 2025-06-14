# Author: Jacek Komorowski, Monika Wysoczanska
# Warsaw University of Technology
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)
import numpy
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import tqdm

from misc.utils import MinkLocParams
from models.model_factory import model_factory
from datasets.dataset_utils import to_spherical


DEBUG = False

def evaluate(model, device, params, log=False):
    # Run evaluation on all eval datasets

    if DEBUG:
        params.eval_database_files = params.eval_database_files[0:1]
        params.eval_query_files = params.eval_query_files[0:1]

    assert len(params.eval_database_files) == len(params.eval_query_files)

    stats = {}
    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, params, database_sets, query_sets)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, params, database_sets, query_sets):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    for set in database_sets:
        database_embeddings.append(get_latent_vectors(model, set, device, params))

    for set in query_sets:
        query_embeddings.append(get_latent_vectors(model, set, device, params, query=True))

    for i in tqdm.tqdm(range(len(query_sets))):
        for j in range(len(query_sets)):
            pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity}
    return stats


def rotate_pc_batch(batch_pc, axis="z", max_angle=90, intensity=False):
    """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    if not intensity:

        B, N, C = batch_pc.shape
        assert C == 3

        rotation_angles = np.ones(B) * 2 * np.pi * max_angle / 360

        cosvals = np.cos(rotation_angles)
        sinvals = np.sin(rotation_angles)

        rotated_batch = np.zeros_like(batch_pc)

        for i in range(B):

            cosval = cosvals[i]
            sinval = sinvals[i]

            if axis == "y":
                rotation_matrix = np.array([[cosval, 0, sinval],
                                            [0, 1, 0],
                                            [-sinval, 0, cosval]])

            elif axis == "x":
                rotation_matrix = np.array([[1, 0, 0],
                                            [0, cosval, -sinval],
                                            [0, sinval, cosval]])
            elif axis == "z":
                rotation_matrix = np.array([[cosval, -sinval, 0],
                                            [sinval, cosval, 0],
                                            [0, 0, 1]])
            else:
                print("axis wrong: ", axis)
                exit(-1)

            rotated_batch[i] = np.dot(batch_pc[i], rotation_matrix)

        return rotated_batch

    else:
        B, N, C = batch_pc.shape
        assert C == 4

        rotation_angles = np.ones(B) * 2 * np.pi * max_angle / 360

        cosvals = np.cos(rotation_angles)
        sinvals = np.sin(rotation_angles)

        rotated_batch = np.zeros_like(batch_pc)

        for i in range(B):

            cosval = cosvals[i]
            sinval = sinvals[i]

            if axis == "y":
                rotation_matrix = np.array([[cosval, 0, sinval, 0],
                                            [0, 1, 0, 0],
                                            [-sinval, 0, cosval, 0],
                                            [0, 0, 0, 1]])

            elif axis == "x":
                rotation_matrix = np.array([[1, 0, 0, 0],
                                            [0, cosval, -sinval, 0],
                                            [0, sinval, cosval, 0],
                                            [0, 0, 0, 1]])
            elif axis == "z":
                rotation_matrix = np.array([[cosval, -sinval, 0, 0],
                                            [sinval, cosval, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
            else:
                print("axis wrong: ", axis)
                exit(-1)

            rotated_batch[i] = np.dot(batch_pc[i], rotation_matrix)

        return rotated_batch


def random_drop_batch(batch_points, drop_angle=60, intensity=False):
    """
    :param batch_points: B*N*3
    :param drop_angle:
    :return: B*N*3。
    """
    if not intensity:
        B, N, C = batch_points.shape
        assert C == 3  # 确保每个点有3个维度（x、y、z）

        droped_batch = np.copy(batch_points)

        for i in range(B):
            start_angle = np.random.random() * 360

            end_angle = (start_angle + drop_angle) % 360

            angle = np.arctan2(droped_batch[i, :, 1], droped_batch[i, :, 0])
            angle = angle * 180 / np.pi
            angle += 180

            if end_angle > start_angle:
                drop_id = np.argwhere((angle > start_angle) & (angle < end_angle)).reshape(-1)
            else:
                drop_id = np.argwhere((angle > start_angle) | (angle < end_angle)).reshape(-1)

            droped_batch[i, drop_id, :] = 0

        return droped_batch

    else:
        B, N, C = batch_points.shape
        assert C == 4  # 确保每个点有3个维度（x、y、z）

        droped_batch = np.copy(batch_points)

        for i in range(B):
            start_angle = np.random.random() * 360

            end_angle = (start_angle + drop_angle) % 360

            angle = np.arctan2(droped_batch[i, :, 1], droped_batch[i, :, 0])
            angle = angle * 180 / np.pi
            angle += 180

            if end_angle > start_angle:
                drop_id = np.argwhere((angle > start_angle) & (angle < end_angle)).reshape(-1)
            else:
                drop_id = np.argwhere((angle > start_angle) | (angle < end_angle)).reshape(-1)

            droped_batch[i, drop_id, :] = 0

        return droped_batch


def rotate_point_cloud(batch_data, intensity=False):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if not intensity:

        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            #rotation_angle = np.random.uniform() * 2 * np.pi
            #-90 to 90
            rotation_angle = (np.random.uniform()*np.pi) - np.pi/2.0
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, -sinval, 0],
                                        [sinval, cosval, 0],
                                        [0, 0, 1]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(
                shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    else:
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            #rotation_angle = np.random.uniform() * 2 * np.pi
            #-90 to 90
            rotation_angle = (np.random.uniform()*np.pi) - np.pi/2.0
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, -sinval, 0, 0],
                                        [sinval, cosval, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(
                shape_pc.reshape((-1, 4)), rotation_matrix)
        return rotated_data

def jitter_point_cloud(batch_data, sigma=0.1, clip=1, intensity=False):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    if not intensity:
        B, N, C = batch_data.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
        jittered_data += batch_data
        return jittered_data
    else:
        B, N, C = batch_data.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C-1), -1*clip, clip)
        batch_data[:, :, :C-1] += jittered_data
        return batch_data


sigma_dict = {'0': 1.5, '1': 3, '2': 4.5, '3': 6, '4': 7.5, '5': 9}
clip_dict = {'0': 3, '1': 6, '2': 9, '3': 12, '4': 15, '5': 18}
index = '0'
def translate_point_cloud(batch_data, sigma=sigma_dict[index], clip=clip_dict[index], intensity=False):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    if not intensity:
        B, N, C = batch_data.shape
        assert(clip > 0)
        tl = np.clip(sigma * np.random.randn(2), -1 * clip, clip)
        tl = np.tile(tl, B*N).reshape(B, N, 2)
        batch_data[:, :, 0:2] += tl

        return batch_data
    else:
        B, N, C = batch_data.shape
        assert (clip > 0)
        tl = np.clip(sigma * np.random.randn(2), -1 * clip, clip)
        tl = np.tile(tl, B * N).reshape(B, N, 2)
        batch_data[:, :, 0:2] += tl

        return batch_data



def load_pc(filename, params, query):
    # Load point cloud, does not apply any transform
    # Returns Nx3 matrix or Nx4 matrix depending on the intensity value
    file_path = os.path.join(params.dataset_folder, filename)

    if params.dataset_name == "USyd":
        pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4])
    elif params.dataset_name == "KITTI":
        pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4])
    elif params.dataset_name == "IntensityOxford":
        pc = np.fromfile(file_path, dtype=np.float64).reshape([-1, 4])
    elif params.dataset_name == "Oxford":
        pc = np.fromfile(file_path, dtype=np.float64).reshape([-1, 3])
        assert pc.size == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)

    # remove intensity for models which are not using it
    if params.model_params.version in ['MinkLoc3D', 'MinkLoc3D-S']:
        pc = pc[:, :3]


    if query:
        intensity = False
        if params.model_params.version in ['MinkLoc3D-SI']:
            intensity = True

        pc = np.expand_dims(pc, axis=0)
        pc = rotate_pc_batch(pc, intensity=intensity)
        # pc = random_drop_batch(pc, intensity=intensity)
        pc = jitter_point_cloud(pc, intensity=intensity)
        # pc = translate_point_cloud(pc, intensity=intensity)
        pc = np.squeeze(pc, axis=0)


    # limit distance
    mask = np.linalg.norm(pc[:, :3], axis=1) >= params.max_distance
    pc[mask] = np.zeros_like(pc[mask])

    assert params.num_points == len(pc)

    # convert to spherical coordinates in -S versions
    if params.model_params.version in ['MinkLoc3D-S', 'MinkLoc3D-SI']:
        pc_s = to_spherical(pc, params.dataset_name)
    else:
        pc_s = pc

    # shuffle points in case they are randomly subsampled later
    # np.random.shuffle(pc)

    pc = torch.tensor(pc, dtype=torch.float)
    assert params.num_points == len(pc)

    pc_s = torch.tensor(pc_s, dtype=torch.float)

    pcs = [pc, pc_s]

    return pcs[0][:, 0:3], pcs[1]

def get_latent_vectors(model, set, device, params, query=False):
    # Adapted from original PointNetVLAD code

    """
    if DEBUG:
        embeddings = torch.randn(len(set), 256)
        return embeddings
    """

    if DEBUG:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    model.eval()
    embeddings_l = []
    # for i, elem_ndx in enumerate(set):
    for elem_ndx in tqdm.tqdm(set):
        filename = 'kitti/00/' + '00' + set[elem_ndx]["query_velo"].split('/')[-1]
        x, x_s = load_pc(filename, params, query)
        # x, x_s = load_pc(set[elem_ndx]["query"], params)
        with torch.no_grad():
            # models without intensity
            if params.model_params.version in ['MinkLoc3D', 'MinkLoc3D-S']:
                coords = ME.utils.sparse_quantize(coordinates=x_s,
                                                  quantization_size=params.model_params.mink_quantization_size)
                bcoords = ME.utils.batched_coordinates([coords]).to(device)
                # Assign a dummy feature equal to 1 to each point
                # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
                feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).to(device)

            # models with intensity - intensity value is averaged over voxel
            elif params.model_params.version in ['MinkLoc3D-I', 'MinkLoc3D-SI']:
                sparse_field = ME.TensorField(features=x_s[:, 3].reshape([-1, 1]),
                                              coordinates=ME.utils.batched_coordinates(
                                                  [x_s[:, :3] / np.array(params.model_params.mink_quantization_size)],
                                                  dtype=torch.int),
                                              quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                              minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED).sparse()
                feats = sparse_field.features.to(device)
                bcoords = sparse_field.coordinates.to(device)

            batch = {'coords': bcoords, 'features': feats}
            batch['plnt_coords'] = x.unsqueeze(dim=0).to(device)
            embedding = model(batch)
            # embedding is (1, 256) tensor
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


def get_recall(m, n, database_vectors, query_vectors, query_sets):
    # based on original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'], stats[database_name]['average_similarity']))
        print(stats[database_name]['ave_recall'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on KITTI dataset')
    parser.add_argument('--config', default='../config/config_kitti.txt',
                        help='Path to configuration file')
    parser.add_argument('--model_config', default='../config/model_config_usyd.txt',
                        help='Path to the model-specific configuration file')
    parser.add_argument('--weights', default='../weights/model_MinkFPN_20240116_2050_final.pth',
                        help='Trained model weights')

    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--log', dest='log', action='store_true')
    parser.set_defaults(log=False)

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('Debug mode: {}'.format(args.debug))
    print('Visualize: {}'.format(args.visualize))
    print('Log search results: {}'.format(args.log))
    print('')

    params = MinkLocParams(args.config, args.model_config)

    # params.eval_database_files = ["/home/hit201/PycharmProjects/PPLoc3D_pnv/benchmark_datasets/kitti_evaluation_database.pickle"]
    # params.eval_query_files = ["/home/hit201/PycharmProjects/PPLoc3D_pnv/benchmark_datasets/kitti_evaluation_query.pickle"]

    params.eval_database_files = [
        "KITTI_00_database_samp10.pickle"]
    params.eval_query_files = [
        "KITTI_00_query_samp10.pickle"]

    params.dataset_folder = "/home/hit201/PycharmProjects/PPLoc3D_pnv/benchmark_datasets/"
    params.dataset_name = 'KITTI'
    params.model_params.mink_quantization_size = [2.5, 2.0, 0.42]
    params.print()
    print('#'*30)
    print('WARNING: Database and query files, paths and quantization from config are overwritten by KITTI specs in evaluate_kitti.py.')
    print('#'*30, '\n')

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    stats = evaluate(model, device, params, args.log)
    print_eval_stats(stats)

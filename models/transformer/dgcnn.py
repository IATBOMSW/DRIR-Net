import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def knn(x, k):
    # Compute the inner product of points
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)  # [b, num, num]

    # Calculate the sum of squares for each point's coordinates
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [b, 1, num]

    # Calculate the squared Euclidean distance between points
    pairwise_distance = -xx - inner
    del inner, x

    # Get the top k indices along the last dimension
    _, idx = pairwise_distance.topk(k=k, dim=-1, largest=False)  # [b, num, k]
    return idx


def get_graph_feature_per_sector(x, k=20, idx=None, cat=True):
    device = x.device
    batch_size = x.size(0)
    num_npoint = x.size(2)
    x = x.view(batch_size, -1, num_npoint)

    if idx is None:
        idx = knn(x, k=k)  # Shape: (batch_size, num_npoint, k)

    # Create indices for each sector
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_npoint  # Shape: (batch_size, 1, 1)
    idx = idx + idx_base  # Shape: (batch_size, num_npoint, k)
    idx = idx.view(-1) # Shape: (batch_size * num_npoint * k)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_npoint, num_dims)
    feature = x.view(batch_size * num_npoint, -1)[idx, :] # (batch_size * num_npoint * k,num_dims)
    feature = feature.view(batch_size, num_npoint, k, num_dims)  # (batch_size, num_npoint, k, num_dims)

    if cat:
        x = x.view(batch_size, num_npoint, 1, num_dims).repeat(1, 1, k, 1)  # [batch_size, num_npoint, k, num_dims]
        feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)  # [batch_size, num_dims*2, num_npoint, k]
    else:
        feature = feature.permute(0, 3, 1, 2)

    return feature

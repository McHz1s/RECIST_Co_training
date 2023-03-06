from collections import abc as container_abc

import numpy as np
import torch
from torch.nn import functional as F


def tensor2prob_map(in_tensor, trivial=False):
    from core.network.utils import sigmoid
    cls_num = in_tensor.size()[1]
    prob_map = sigmoid(in_tensor) if cls_num == 1 else F.softmax(in_tensor, dim=1)
    if cls_num == 1 and trivial:
        prob_map = torch.cat([prob_map, 1 - prob_map], dim=1)
    return prob_map


def feature_transform(feature, t_matrix, device=None, dtype='float'):
    if device is not None:
        feature, t_matrix = tuple(map(lambda x: to_device(x, device, dtype), [feature, t_matrix]))
    grid = F.affine_grid(t_matrix, feature.size(), align_corners=True)
    re = F.grid_sample(feature, grid, align_corners=True)
    return re


def make_transform_matrix(mag):
    basic_matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
    shift, scale_list, shift_index_list, scale_index_list = \
        [-mag, mag], [1 - mag, 1 / (1 - mag)], [(0, 2), (1, 2)], [(0, 0), (1, 1)]
    transform_matrix_list = [basic_matrix]
    for op, scale in zip(shift, scale_list):
        for shift_index, scale_index in zip(shift_index_list, scale_index_list):
            plus = np.zeros_like(basic_matrix)
            plus[shift_index] = op * 1
            transform_matrix_list.append(basic_matrix + plus)
            scale_matrix = basic_matrix.copy()
            scale_matrix[scale_index] *= scale
            transform_matrix_list.append(scale_matrix)

    return transform_matrix_list


def petrificus_totalus(inp):
    if isinstance(inp, container_abc.Mapping):
        return {key: petrificus_totalus(inp[key]) for key in inp}
    if isinstance(inp, list):
        return [petrificus_totalus(item) for item in inp]
    if not isinstance(inp, torch.Tensor):
        return inp
    inp = inp.detach()
    if inp.device.type == 'cuda':
        inp = inp.cpu()
    return inp.numpy()


def to_device(inp, device, to_type=None):
    if isinstance(inp, container_abc.Mapping):
        return {key: to_device(inp[key], device, to_type) for key in inp}
    if isinstance(inp, container_abc.Sequence):
        return [to_device(item, device, to_type) for item in inp]
    if isinstance(inp, np.ndarray):
        inp = torch.as_tensor(inp)
    if to_type is None or to_type == 'identity':
        if device == 'identity' or device is None:
            return inp
        return inp.to(device)
    return inp.to(device).__getattribute__(to_type)()

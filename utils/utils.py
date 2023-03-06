import json
import math
import os
import yaml
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import torchvision
from tensorboard.backend.event_processing import event_accumulator

INT_MAX = 0x3f3f3f


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)

    return yaml.load(stream, OrderedLoader)


def setup_device(gpus, mannul=False):
    if not mannul:
        if gpus is None:
            device = torch.device("cpu")
            return device
        if os.environ['RANK'] in ['-1', '0']:
            print('gpus', gpus)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus  # "0, 5, 6"
    if gpus is None or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def get_func_keyword(func):
    """
    :param func: function
    :return: sequence(tuple) of function keyword arguments.
    """
    return func.__code__.co_varnames


def get_random_pt_from_mask(mask):
    index = np.argwhere(mask != 0)
    pt = random.randint(0, len(index))
    return np.flip(index[pt])


def rescale_dets(detections, ratios):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs /= ratios[1]
    ys /= ratios[0]


def rescale_pts(detections, ratios):
    xs, ys = detections[..., 5:13:2], detections[..., 6:13:2]
    xs /= ratios[1]
    ys /= ratios[0]


def eval_bn(model, names):
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
            continue
        for n in names:
            if n in name:
                m.eval()


def nms_with_bboxes(detections, nms_threshold):
    # detections: [batch_size, 100, 5]
    post_detections = []
    post_pts = []
    for i in range(len(detections)):
        dets = detections[i]
        # reject detections with negative scores
        keep_inds = (dets[:, 4] > 0)
        dets = dets[keep_inds]

        boxes = dets[:, :4]
        scores = dets[:, 4]
        extreme_points = dets[:, 5:13]
        keep_inds = torchvision.ops.nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_inds]
        scores = scores[keep_inds]
        extreme_points = extreme_points[keep_inds]
        scores = torch.unsqueeze(scores, dim=1)
        dets = torch.cat((boxes, scores), dim=1)
        dets = dets.data.cpu().numpy()
        extreme_points = extreme_points.data.cpu().numpy()
        post_detections.append(dets)
        post_pts.append(extreme_points)
    return post_detections, post_pts


def read_tensorboard(tensorboard_path):
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    return ea


def load_json(json_path):
    with open(json_path, 'r') as load_f:
        re_json = json.load(load_f)
    return re_json


def save_json(json_path, json_output):
    with open(json_path, 'w') as output_json_file:
        json.dump(json_output.copy(), output_json_file)


def get_worker_id():
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        return worker_info.id
    return 0


def custom_load_pretrained_dict(model_state_dict, pretrained_state_dict, strict=True):
    model_need_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
    if not strict:
        model_need_dict_free = {}
        for key, item in model_need_dict.items():
            if item.size() == model_state_dict[key].size():
                model_need_dict_free[key] = item
        model_need_dict = model_need_dict_free
    model_state_dict.update(model_need_dict)


def median_ci(data, confidence=0.95):
    data1 = sorted(data)
    n = len(data1)
    ll = 0.5 * n - 0.98 * math.sqrt(n)
    ul = 1 + 0.5 * n + 0.98 * math.sqrt(n)
    low = data1[math.ceil(ll) - 1]
    up = data1[math.floor(ul) - 1]
    return low, up


def interval_judge(value, interval_list):
    """

    Args:
        interval_list: list of sub interval expressed by list: [[a,b], [c,d]....]
    Returns:
        True if value in any sub interval
    """
    if not interval_list:
        return False
    return any(list(map(lambda interval: interval[1] >= value >= interval[0], interval_list)))

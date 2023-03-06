from utils.my_containers import Constructor

network_register = Constructor()

from .wrapper.muti_parallel_model import MutiParallelModel
from .wrapper.segmenation_model_2d import NormalSeg2D


def get_network(cfg):
    network = network_register.build_with_cfg(cfg)
    return network

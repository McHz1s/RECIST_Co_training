from utils.my_containers import Register

POSTPROCESSOR = Register()
from .wss.muti_model_post_process import MultiModelBinarySemanticSegmentationPostProcess
from .wss.binary_segmentation_post_process import BinarySemanticSegmentationPostProcess


def get_post_process(post_precess_cfg):
    return POSTPROCESSOR[post_precess_cfg.name](post_precess_cfg)

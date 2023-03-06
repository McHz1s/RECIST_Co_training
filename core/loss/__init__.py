import importlib

from utils.loggers import LoggerManager
from .atom_loss import *
from .loss import ATOMLOSS


def get_loss_function(loss_name, loss_cfg):
    path = 'core.loss.loss'
    loss_module = importlib.import_module(path)
    loss = getattr(loss_module, loss_name)(loss_cfg)
    logger = LoggerManager().get_logger('Loss')
    logger("Using {} with {} params".format(loss_name, loss_cfg))
    return loss

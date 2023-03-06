import abc
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.loggers import LoggerManager
from utils.utils import custom_load_pretrained_dict


class BasicWrapper(nn.Module):
    _network_training = 0
    _network_validing = 1
    _network_testing = 2

    def __init__(self, cfg_network_part):
        super(BasicWrapper, self).__init__()
        self.cfg = cfg_network_part
        self.logger = LoggerManager().get_logger('Network')
        self.network_status = self.init_network_status()

    def switch_network_status(self, to_status='training'):
        self.network_status = self.__getattribute__(f'_network_{to_status}')
        return self.network_status

    def init_network_status(self):
        return self.switch_network_status('training')

    def load_pretrained_params(self, pretrained_model_path, strict, pretrained_state=None):
        if pretrained_state is None:
            assert pretrained_model_path is not None
            with open(pretrained_model_path, "rb") as f:
                state_dict = torch.load(f, map_location='cpu')
        else:
            state_dict = pretrained_state
        new_pretrained_state_dict = OrderedDict()
        if "convert_pretrained" not in self.cfg or self.cfg.convert_pretrained is None:
            pretrained_dict = state_dict["network"]
        else:
            pretrained_dict = self.__getattribute__(self.cfg.convert_pretrained)(state_dict)
        for k, v in pretrained_dict.items():
            name = '.'.join(k.split('.')[1:]) if k.startswith('module.') else k  # remove 'module.'
            new_pretrained_state_dict[name] = v
        model_dict = self.state_dict()
        custom_load_pretrained_dict(model_dict, new_pretrained_state_dict, strict=strict)
        self.load_state_dict(model_dict, strict=True)

    @abc.abstractmethod
    def convert_pretrained(self, pretrained_dict):
        pass

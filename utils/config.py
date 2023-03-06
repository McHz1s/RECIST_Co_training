from collections import abc as container_abc

import yaml

from utils.my_containers import ObjDict
from utils.utils import ordered_load
from .mmcv_config import Config as pyConfig


def load_yaml(config_path):
    with open(config_path, encoding='utf-8') as fp:
        cfg = ordered_load(fp, yaml.SafeLoader)
    return ObjDict(cfg)


def load_py(config_path):
    return ObjDict(pyConfig.fromfile(config_path))


class ConfigConstructor(object):
    def __init__(self, config_path):
        self.suffix2loadMethods = {'.yaml': load_yaml, '.yml': load_yaml, '.py': load_py}
        self.inherit_tree = {}
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        for suffix, m in self.suffix2loadMethods.items():
            if suffix in config_path:
                return ObjDict(m(config_path)).transform()
        raise NotImplementedError

    def config_inherit(self, base_config_path_list):
        if isinstance(base_config_path_list, str):
            base_config_path_list = [base_config_path_list]
        base_config = ObjDict()
        for base_cfg_path in base_config_path_list:
            self.cfg_update(base_config, ConfigConstructor(base_cfg_path).get_config())
        return base_config

    def construct_config(self, config_dict):
        base_config = ObjDict()
        for key, value in config_dict.items():
            if key == '_Base_Config':
                base_config = self.config_inherit(config_dict['_Base_Config'])
            elif isinstance(value, container_abc.Mapping):
                config_dict[key] = self.construct_config(value)
        self.cfg_update(base_config, config_dict)
        if '_Base_Config' in base_config:
            base_config.pop('_Base_Config')
        return base_config

    def get_config(self):
        cfg = self.construct_config(self.config)
        return cfg

    def cfg_update(self, base_cfg, new_cfg):
        for key, value in new_cfg.items():
            if key not in base_cfg:
                base_cfg[key] = value
            elif isinstance(value, container_abc.Mapping):
                if 'name' in value and value['name'] != base_cfg[key].get('name', None):
                    base_cfg[key] = new_cfg[key]
                else:
                    self.cfg_update(base_cfg[key], new_cfg[key])
            else:
                base_cfg[key] = new_cfg[key]

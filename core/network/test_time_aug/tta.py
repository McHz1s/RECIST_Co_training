from collections import abc as container_abc

from core.network.test_time_aug.transform import *
from utils.my_containers import deep_dict_mean
from . import TTA_REGISTER


class TTA(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def symmetric_aug(self, func):
        trans = func

        def r_trans(output_dict):
            for key, value in output_dict.items():
                if isinstance(value, container_abc.Mapping):
                    r_trans(output_dict[key])
                if key != self.key_word:
                    continue
                if self.key_word == key:
                    output_dict[key] = func(value)
            return output_dict
        return trans, r_trans

    def yield_aug(self):
        for aug in self.cfg.aug_list:
            yield aug, self.__getattribute__(aug)()


@TTA_REGISTER.register
class SegTTA(TTA):
    def __init__(self, *args, **kwargs):
        super(SegTTA, self).__init__(*args, **kwargs)
        self.key_word = 'prob_map'

    def hflip_aug(self):
        return self.symmetric_aug(hflip)

    def vflip_aug(self):
        return self.symmetric_aug(vflip)

    def mean(self, output_dict_list):
        re_dict = deep_dict_mean(output_dict_list, key_pool='prob_map')
        return re_dict

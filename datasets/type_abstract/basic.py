import torch.utils.data as data

from datasets.utils.augmentation import AugmentationSequenceManager
from datasets.utils.calculate_reverse_para import ReverseTransformCalulateManager
from datasets.utils.normalize import NormalizationManager
from datasets.utils.reverse import ReverseTransformManager
from datasets.utils.reverse_normalize import ReverseNormalizationManager


class BasicDataset(data.Dataset):
    def __init__(self, data_cfg):
        """
        basic dataset implementation
        rewrite getitem
        """
        self.data_cfg = data_cfg
        self.init_hook_fn()

    @property
    def hookManager_mapping(self):
        return {'augmentation_seq_hook': AugmentationSequenceManager,
                'normalize_fn_hook': NormalizationManager,
                'reverse_normalize_fn_hook': ReverseNormalizationManager,
                'calculate_reverse_transform_meta_hook': ReverseTransformCalulateManager,
                'reverse_transform_fn_hook': ReverseTransformManager}

    def init_hook_fn(self):
        for hook_name, hook_funcManager in self.hookManager_mapping.items():
            hook_func_setting = self.data_cfg.get(hook_name, {})
            if hook_func_setting != {}:
                self.__setattr__(hook_name, hook_funcManager(hook_func_setting)(hook_func_setting['name']))
            else:
                self.__setattr__(hook_name, None)

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        re_dict = None
        idx_ = idx
        while re_dict is None:
            re_dict = self.get_valid_item(idx_)
            idx_ = (idx_ + 1) % self.__len__()
        return re_dict

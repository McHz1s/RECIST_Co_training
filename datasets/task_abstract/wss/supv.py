import random
from functools import partial

import numpy as np
from imgaug.augmentables import SegmentationMapsOnImage

from datasets import taskAbsRegister


@taskAbsRegister.register
class FullSupv(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_mode = cfg.get('run_mode', 'train')
        self.index_aug_seq = None

    @property
    def index_info_need(self):
        return self.__getattribute__(f'{self.run_mode}_index_info_need')

    @property
    def train_index_info_need(self):
        return ['img', 'gt_mask', 'ori_shape']

    @property
    def valid_index_info_need(self):
        return self.train_index_info_need + ['image_name']

    @property
    def test_index_info_need(self):
        return ['img', 'image_name', 'ori_shape']

    @property
    def eval_index_info_need(self):
        return ['image_name', 'gt_mask']

    def is_valid_aug(self, mask_aug, gt_mask):
        if np.max(gt_mask) == 0 or self.run_mode != ' train':
            return True
        return random.random() < 0.5 or np.max(mask_aug) != 0

    def get_valid_item(self, index):
        index_info = self.get_index_info(index)

        if self.run_mode == 'eval':
            return {'img_name': index_info['image_name'], 'gt_mask': index_info['gt_mask']}

        # normalize
        img_window_norm = self.normalize_fn_hook(index_info['img']).squeeze()

        aug_seq = self.augmentation_seq_hook.to_deterministic()
        gt_mask = index_info['gt_mask']
        while True:
            # augment
            aug_seq.to_deterministic()
            segmap = SegmentationMapsOnImage(gt_mask, shape=index_info['ori_shape'])
            img_aug, mask_aug = aug_seq(image=img_window_norm, segmentation_maps=segmap)
            mask_aug = mask_aug.get_arr().astype(np.uint8)
            mask_aug = np.expand_dims(mask_aug, axis=0)
            if self.is_valid_aug(mask_aug, gt_mask):
                break

        # normalize
        inp = np.expand_dims(img_aug.copy(), axis=0)

        batch_data_dict = {'img': inp, 'supv_mask': mask_aug}
        if self.run_mode in ['valid', 'test']:
            batch_data_dict['reverse_transform'] = partial(self.reverse_transform_fn_hook,
                                                           ori_shape=index_info['ori_shape'])
            batch_data_dict['img_name'] = index_info['image_name']
        if self.run_mode == 'test':
            batch_data_dict['vis_img_transform'] = partial(self.reverse_normalize_fn_hook,
                                                           ori_shape=index_info['ori_shape'])
        return batch_data_dict

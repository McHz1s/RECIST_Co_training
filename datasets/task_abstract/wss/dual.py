from functools import partial

import numpy as np
from imgaug import SegmentationMapsOnImage

from datasets import taskAbsRegister
from datasets.task_abstract.wss.supv import FullSupv
from datasets.utils.augmentation import AugmentationSequenceManager
from datasets.utils.make_label import MaskMaker
from datasets.utils.utils import line_intersection
from visualizer.visualizers import vis_img, vis_real_recist


@taskAbsRegister.register
class DualWSS(FullSupv):
    def __init__(self, cfg):
        super(DualWSS, self).__init__(cfg)
        self.geometry_mask_name_list = self.cfg.get('geometry_mask_name_list', ['diamond', 'circle'])
        self.mask_post_process = self.cfg.get('mask_post_process', False)
        self.post_th = self.cfg.get('post_th', 50)
        if self.cfg.dual_mask_construct_way == 'from_two_stage_construct':
            aug_seq_manager = AugmentationSequenceManager
            self.recist_points_augmentation_hook = \
                aug_seq_manager(self.cfg.recist_points_augmentation_hook)(self.cfg.recist_points_augmentation_hook.name)
            self.dual_mask_augmentation_hook = \
                aug_seq_manager(self.cfg.dual_mask_augmentation_hook)(self.cfg.dual_mask_augmentation_hook.name)

    @property
    def train_index_info_need(self):
        return ['img', 'ori_shape', 'recist_pts_list']

    @property
    def valid_index_info_need(self):
        return self.train_index_info_need + ['image_name']

    @property
    def analyse_index_info_need(self):
        return self.valid_index_info_need

    def from_ori_recist(self, img_window_norm, ori_recist_pts_list):
        ori_shape = img_window_norm.shape[:2]
        # mask
        mask_maker = MaskMaker(ori_recist_pts_list, ori_shape)
        mask_dict = mask_maker.recist2geometry((1,) + ori_shape, self.geometry_mask_name_list)

        # augment
        aug_seq = self.index_aug_seq
        img_aug = aug_seq(image=img_window_norm)
        for mask_name, pseudo_mask in mask_dict.items():
            segmap = SegmentationMapsOnImage(pseudo_mask[0], shape=ori_shape)
            img_aug, mask_aug = aug_seq(image=img_window_norm, segmentation_maps=segmap)
            mask_aug = mask_aug.get_arr().astype(np.uint8)
            mask_aug = np.expand_dims(mask_aug, axis=0)
            mask_dict[mask_name] = mask_aug

        if len(self.geometry_mask_name_list) > 1:
            shield_mask = mask_maker.make_shield(*list(mask_dict.values()))
        else:
            shield_mask = np.zeros_like(list(mask_dict.values())[0])

        return img_aug, mask_dict, shield_mask, ori_recist_pts_list

    def get_valid_item(self, index):
        self.index_aug_seq = self.augmentation_seq_hook.to_deterministic()
        index_info = self.get_index_info(index)

        if self.run_mode == 'eval':
            return {'img_name': index_info['image_name'], 'gt_mask': index_info['gt_mask'],
                    'gt_recist_pts': np.array(index_info['recist_pts_list'])}

        ori_recist_pts_list = index_info['recist_pts_list']

        img = index_info['img']
        # clear dirty data
        if self.run_mode == 'train' and self.cfg.get('remove_dirty_recist', True):
            pts_number = len(ori_recist_pts_list)
            for i in range(len(ori_recist_pts_list)):
                recist_points = ori_recist_pts_list[i]
                if np.any(recist_points < 0):
                    del ori_recist_pts_list[i]
                    continue
                inter = line_intersection(recist_points[:2], recist_points[2:])
                if any(inter <= 0):
                    del ori_recist_pts_list[i]
            if pts_number != 0 and len(ori_recist_pts_list) == 0:
                return None

        # normalize
        img_window_norm = self.normalize_fn_hook(index_info['img'])
        index_info['img_window_norm'] = img_window_norm

        img_aug, mask_dict, shield_mask, kps_aug_list = self.__getattribute__(self.cfg.dual_mask_construct_way)(
            img_window_norm, ori_recist_pts_list)
        if img_aug is None:
            return None
        if len(img_aug.shape) == 2:
            img_aug = np.expand_dims(img_aug, axis=-1)
        inp = np.transpose(img_aug, [2, 0, 1])

        batch_data_dict = {'img': inp.copy(), 'psuedo_mask_dict': mask_dict, 'shield_mask': shield_mask, 'id': index}
        reverse_meta = self.calculate_reverse_transform_meta_hook(img_window_norm) if \
            self.calculate_reverse_transform_meta_hook is not None else {}
        img_meta = {'input_size': inp.shape[:2], 'ori_shape': img.shape[:2]}
        img_meta.update(reverse_meta)
        batch_data_dict['img_metas'] = img_meta
        if self.run_mode in ['valid', 'test']:
            batch_data_dict['reverse_transform'] = partial(self.reverse_transform_fn_hook, **reverse_meta)
            batch_data_dict['img_name'] = index_info['image_name']
        if self.data_cfg.get('need_visualization', False):
            batch_data_dict['visualization'] = {}
            rn_ori_img = self.reverse_normalize_fn_hook(img_window_norm)
            rn_aug_img = self.reverse_normalize_fn_hook(img_aug)
            batch_data_dict['visualization']['ori_img'] = rn_ori_img
            batch_data_dict['visualization']['aug_img'] = rn_aug_img
            batch_data_dict['visualization']['aug_recist_pts'] = kps_aug_list
            batch_data_dict['visualization']['ori_recist_pts'] = ori_recist_pts_list
            batch_data_dict['visualization']['ori_img_ori_recist'] = vis_real_recist(vis_img(rn_ori_img),
                                                                                     ori_recist_pts_list)
            batch_data_dict['visualization']['aug_img_aug_recist'] = vis_real_recist((rn_aug_img),
                                                                                     kps_aug_list)
        return batch_data_dict

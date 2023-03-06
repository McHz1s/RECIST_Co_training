import os

import cv2
import imgaug.augmenters as iaa
import numpy as np

from datasets import instanceRegister
from datasets.type_abstract.coco_format import DatasetCOCOFormat
from datasets.utils.calculate_reverse_para import calculate_paddings
from utils.bbox_operator import BBoxes
from utils.mask_operator import rle2mask, mask_union
from utils.tensor_operator import petrificus_totalus


@instanceRegister.register
class RECIST2DBasic(DatasetCOCOFormat):
    def __init__(self, data_cfg):
        super(RECIST2DBasic, self).__init__(data_cfg)

    def get_index_info(self, index):
        img_id, annos = self.process_info(index)
        index_info = self.read_original_data(img_id, annos)
        return index_info

    def read_original_data(self, img_id, annos):
        index_info = {}
        for need in self.index_info_need:
            index_info[need] = self.__getattribute__(f'read_{need}')(img_id, annos)
        return index_info

    def read_img(self, img_id, annos):
        img_name = self.read_image_name(img_id, annos)
        img_path = os.path.join(self.img_path, img_name)
        img = np.load(img_path)
        return img

    def read_bbox(self, img_id, annos):
        bboxes = [[bbox for bbox in obj['bbox']] for obj in annos]
        bbox = BBoxes(bboxes, mode='xywh')
        return bbox

    def read_image_name(self, img_id, annos):
        try:
            img_name = self.coco.imgs[img_id]['filename']
        except:
            img_name = self.coco.imgs[img_id]['file_name']
        return img_name

    def read_recist_pts_list(self, img_id, anno_list):
        recist_pts_list = [np.array([ex for ex in obj['recist_points']]) for obj in anno_list]
        return recist_pts_list

    def read_cls_ids(self, img_id, anno_list):
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno_list]
        return cls_ids

    def read_gt_mask(self, img_id, anno_list):
        gt_mask = []
        for anno in anno_list:
            mask_rle, width, height = tuple(map(anno.get, ['segmentation', 'width', 'height']))
            mask = rle2mask(mask_rle, (height, width))
            gt_mask.append(mask.astype(np.bool))
        gt_mask = mask_union(gt_mask)
        return gt_mask

    def read_ori_shape(self, img_id, anno_list):
        shape = tuple(map(self.coco.imgs[img_id].get, ['height', 'width']))
        return shape

    def read_spacing(self, img_id, anno_list):
        return anno_list[0]['spacing']

    def vis_img_transform(self, img, reverse_meta):
        img = petrificus_totalus(img)
        img = np.transpose(img, [1, 2, 0])
        img = self.reverse_transform_fn_hook(img, interpolation=cv2.INTER_LINEAR, **reverse_meta)
        reverse_norm_img = self.reverse_normalize_fn_hook(img)
        return reverse_norm_img

    def make_visualization(self, img_window_norm, input_size):
        img_norm = self.reverse_normalize_fn_hook(img_window_norm)
        seq = iaa.Sequential([iaa.PadToFixedSize(width=input_size[1], height=input_size[0], position='center')])
        pad_amount = calculate_paddings(*img_window_norm.shape[:2], *input_size)


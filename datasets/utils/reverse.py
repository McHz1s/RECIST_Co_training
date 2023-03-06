import cv2
import numpy as np


class ReverseTransformManager(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, name):
        return self.__getattribute__(name)

    @staticmethod
    def basic_resize_mask_transform(mask, ori_shape, interpolation=cv2.INTER_NEAREST, *args, **kwargs):
        return cv2.resize(mask.astype(np.uint8), ori_shape[::-1],
                          interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def basic_resize_and_crop(in_array, ori_shape, pad_amount, interpolation=cv2.INTER_NEAREST, *args, **kwargs):
        array_shape = in_array.shape
        crop_array = in_array[pad_amount[0]: array_shape[0] - pad_amount[2], pad_amount[1]: array_shape[1] - pad_amount[3]]
        ori_array = cv2.resize(crop_array, ori_shape[::-1], interpolation=interpolation)
        return ori_array

    @staticmethod
    def basic_threshold_resize_and_crop(in_array, ori_shape, pad_amount, size_range, interpolation=cv2.INTER_NEAREST,
                                        *args, **kwargs):
        mask_shape = in_array.shape
        crop_array = in_array[pad_amount[0]: mask_shape[0] - pad_amount[2],
                     pad_amount[1]: mask_shape[1] - pad_amount[3]]
        if not size_range[0] <= max(ori_shape[:2]) <= size_range[1]:
            ori_array = cv2.resize(crop_array, ori_shape[::-1],
                                   interpolation=interpolation)
        else:
            ori_array = crop_array.squeeze()
        return ori_array

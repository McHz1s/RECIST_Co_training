import numpy as np

from utils.CT_preprocess import window_level_normalize, min_max_normalize, z_score_normalize


class NormalizationManager(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, name):
        return self.__getattribute__(name)

    def window_level_normalize(self, img, *args, **kwargs):
        return window_level_normalize(img, self.cfg.level, self.cfg.window,
                                      normalize_func=self.cfg.get('normalize_func', 'min_max_normalize'))

    def min_max_normalize(self, img, *args, **kwargs):
        return min_max_normalize(img, self.cfg.get('min', None), self.cfg.get('max', None))

    def z_score_normalize(self, img, *args, **kwargs):
        return z_score_normalize(img, self.cfg.get('mean', None), self.cfg.get('max', None))

    def slice_repeats_muti_view_normalize(self, img, *args, **kwargs):
        img = img.copy()
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, axis=-1, repeats=self.cfg.slice_repeats)
        return self.muti_view_normalize(img)

    def muti_view_normalize(self, img):
        img = img.copy()
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        norm_img_list = []
        for level, window in zip(self.cfg.level_list, self.cfg.window_list):
            norm_img = window_level_normalize(img, level, window)
            norm_img_list.append(norm_img)
        img = np.concatenate(norm_img_list, axis=-1)
        if self.cfg.get('channel_shuffle', False):
            img = np.transpose(img, [2, 0, 1])
            np.random.shuffle(img)
            img = np.transpose(img, [1, 2, 0])
        return img.copy()

    def shuffle_muti_view_normalize(self, img):
        img = self.muti_view_normalize(img)
        np.random.shuffle(img)
        return img.copy()


class ReverseNormalizationManager(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, name):
        return self.__getattribute__(name)

    def reverse_window_level_normalize(self, img, *args, **kwargs):
        return img.copy()

    def reverse_slice_repeats_muti_view_normalize(self, img, *args, **kwargs):
        return img[..., self.cfg.vis_img_slice].copy()

    def reverse_muti_view_normalize(self, img, *args, **kwargs):
        return img[..., self.cfg.vis_img_slice].copy()

import cv2
import imgaug.augmenters as iaa

from utils.my_containers import ObjDict


class AugmentationSequenceManager(object):
    def __init__(self, aug_cfg):
        self.cfg = aug_cfg

    def __call__(self, name):
        if isinstance(name, str):
            return self.__getattribute__(name)()
        aug_seq_list = []
        for each_name in name:
            aug_seq = self.__getattribute__(each_name)()
            aug_seq_list.append(aug_seq)
        return iaa.Sequential(aug_seq_list)

    def basic_identity(self):
        return iaa.Sequential([iaa.Noop()])

    def basic_resize(self):
        return iaa.Sequential([
            iaa.Resize({"height": self.cfg.input_size[0], "width": self.cfg.input_size[1]})])

    def basic_keep_aspect_resize(self):
        return iaa.Sequential([iaa.Resize({"shorter-side": "keep-aspect-ratio",
                                           "longer-side": self.cfg.input_size[0]}, interpolation=cv2.INTER_LINEAR),
                               iaa.PadToFixedSize(width=self.cfg.input_size[1], height=self.cfg.input_size[0],
                                                  position='center')])

    def classical_RECIST2D_augmentation(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        default_value_name = ['crop_ratio', 'pad_ratio',
                              'x_size_scale', 'y_size_scale',
                              'x_translate_percent', 'y_translate_percent',
                              'rotate_range']
        default_value = [0.1, 0.1,
                         [0.8, 1.3], [0.8, 1.3],
                         [-0.1, 0.1], [-0.1, 0.1],
                         [-30, 30]]
        default_aug_cfg_dict = ObjDict(dict(zip(default_value_name, default_value)))
        default_aug_cfg_dict.update(self.cfg)
        aug_para = default_aug_cfg_dict

        aug_seq = iaa.Sequential([
            iaa.Resize({"height": aug_para.input_size[0],
                        "width": aug_para.input_size[1]}),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.CropAndPad(percent=(-aug_para.crop_ratio, aug_para.pad_ratio), pad_mode="constant"),
            sometimes(iaa.Affine(
                scale={"x": aug_para.x_size_scale, "y": aug_para.y_size_scale},
                translate_percent={"x": aug_para.x_translate_percent, "y": aug_para.y_translate_percent},
                rotate=tuple(aug_para.rotate_range),
                mode="constant"
            )),
        ])
        return aug_seq

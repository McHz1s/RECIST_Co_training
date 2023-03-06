import cv2
import imgaug.augmenters as iaa


def calculate_paddings(height_image, width_image,
                       height_min, width_min, pad_xs_i=0.5, pad_ys_i=0.5):
    # default is center paddings
    pad_top = 0
    pad_right = 0
    pad_bottom = 0
    pad_left = 0

    if width_min is not None and width_image < width_min:
        pad_total_x = width_min - width_image
        pad_left = int((1 - pad_xs_i) * pad_total_x)
        pad_right = pad_total_x - pad_left

    if height_min is not None and height_image < height_min:
        pad_total_y = height_min - height_image
        pad_top = int((1 - pad_ys_i) * pad_total_y)
        pad_bottom = pad_total_y - pad_top

    # return pad_top, pad_right, pad_bottom, pad_left
    return pad_top, pad_left, pad_bottom, pad_right


class ReverseTransformCalulateManager(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, name):
        return self.__getattribute__(name)

    def basic_resize_mask_transform(self, ori_img, *args, **kwargs):
        return {'ori_shape': ori_img.shape[:2]}

    def basic_resize_and_crop(self, ori_img, *args, **kwargs):
        ori_shape = ori_img.shape[:2]
        resize_img = iaa.Resize({"shorter-side": "keep-aspect-ratio",
                                 "longer-side": self.cfg.input_size[0]})(image=ori_img)
        pad_amount = iaa.compute_paddings_to_reach_aspect_ratio(resize_img, aspect_ratio=1)
        return {'pad_amount': pad_amount, 'ori_shape': ori_shape}

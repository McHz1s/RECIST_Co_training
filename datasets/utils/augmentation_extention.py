import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np


class ResizeToThreshold(iaa.Resize):
    def __init__(self, size_threshold, direction='up', *args, **kwargs):
        super(ResizeToThreshold, self).__init__(*args, **kwargs)
        self.size_range = size_threshold

        def _if_need_resize_func(input_size):
            long_side = max(input_size)
            if direction == 'up':
                return long_side < self.size_range
            if direction == 'down':
                return long_side > self.size_range

        self.if_need_resize_func = _if_need_resize_func

    # Added in 0.4.0.
    def _augment_images_by_samples(self, images, samples):
        input_was_array = False
        input_dtype = None
        if ia.is_np_array(images):
            input_was_array = True
            input_dtype = images.dtype

        samples_a, samples_b, samples_ip = samples
        result = []
        for i, image in enumerate(images):
            if not self.if_need_resize_func(image.shape):
                image_rs = image
            else:
                h, w = self._compute_height_width(image.shape, samples_a[i],
                                                  samples_b[i], self.size_order)
                image_rs = ia.imresize_single_image(image, (h, w),
                                                    interpolation=samples_ip[i])
            result.append(image_rs)

        if input_was_array:
            all_same_size = (len({image.shape for image in result}) == 1)
            if all_same_size:
                result = np.array(result, dtype=input_dtype)

        return result

    # Added in 0.4.0.
    def _augment_maps_by_samples(self, augmentables, arr_attr_name, samples):
        result = []
        samples_h, samples_w, samples_ip = samples

        for i, augmentable in enumerate(augmentables):
            arr = getattr(augmentable, arr_attr_name)
            arr_shape = arr.shape
            if not self.if_need_resize_func(arr_shape):
                augmentable_resize = augmentable
            else:
                img_shape = augmentable.shape
                h_img, w_img = self._compute_height_width(
                    img_shape, samples_h[i], samples_w[i], self.size_order)
                h = int(np.round(h_img * (arr_shape[0] / img_shape[0])))
                w = int(np.round(w_img * (arr_shape[1] / img_shape[1])))
                h = max(h, 1)
                w = max(w, 1)
                if samples_ip[0] is not None:
                    # TODO change this for heatmaps to always have cubic or
                    #      automatic interpolation?
                    augmentable_resize = augmentable.resize(
                        (h, w), interpolation=samples_ip[i])
                else:
                    augmentable_resize = augmentable.resize((h, w))
                augmentable_resize.shape = (h_img, w_img) + img_shape[2:]
            result.append(augmentable_resize)

        return result

    # Added in 0.4.0.
    def _augment_keypoints_by_samples(self, kpsois, samples):
        result = []
        samples_a, samples_b, _samples_ip = samples
        for i, kpsoi in enumerate(kpsois):
            if not self.if_need_resize_func(kpsoi.shape):
                keypoints_on_image_rs = kpsoi
            else:
                h, w = self._compute_height_width(
                    kpsoi.shape, samples_a[i], samples_b[i], self.size_order)
                new_shape = (h, w) + kpsoi.shape[2:]
                keypoints_on_image_rs = kpsoi.on_(new_shape)

            result.append(keypoints_on_image_rs)

        return result

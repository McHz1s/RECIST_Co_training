import re

from utils.tensor_operator import petrificus_totalus
from visualizer.visualizers import *


class Visualizer(object):
    def __init__(self, img_name, raw_img, writer, cfg):
        self.img_name = img_name
        self.raw_img = raw_img
        self.color_img = self.cvt_color(raw_img)
        self.writer = writer
        self.to_writer_dict = {}
        self.vis_pool = []
        self.vis_mapper = {'mask': vis_mask_outline,
                           'hm': vis_many_hm,
                           'heatmap': vis_many_hm,
                           'recist': vis_real_recist,
                           'bbox': vis_bbox,
                           'img': vis_img,
                           'image': vis_img}
        self.cfg = cfg
        if self.cfg.get('mask_vis_method', 'vis_mask_outline') != 'vis_mask_outline':
            self.vis_mapper['mask'] = vis_many_mask

    def vis_mapping(self, name):
        """

        Args:
            name: trouble's name

        Returns:
            a function to show this trouble
        """
        for shower_name, shower in self.vis_mapper.items():
            if re.search(shower_name, name) is not None:
                return shower
        raise ValueError(f'No visualizer for {name}')

    def vis(self, affiliation_dict, is_plt_show=False,
            is_add_writer=True, writer_title=None, suffix=None, **writer_para):
        title = writer_title if writer_title is not None else self.img_name
        self.writer.add_image(f'{title}/origin', self.color_img, dataformats='HWC', **writer_para)
        for affiliation_name, affiliation in affiliation_dict.items():
            show_way = self.vis_mapping(affiliation_name)
            show = show_way(self.color_img, petrificus_totalus(affiliation))
            if is_plt_show:
                plt_show(show)
            if is_add_writer:
                title = writer_title if writer_title is not None else self.img_name
                if suffix is not None:
                    title = f'{title}_{suffix}'
                title = f'{title}/{affiliation_name}'
                self.writer.add_image(title, show, dataformats='HWC', **writer_para)

    @staticmethod
    def cvt_color(raw_img):
        color_img = raw_img.copy()
        color_img = petrificus_totalus(color_img)
        if np.max(color_img) <= 1:
            color_img *= 255
        color_img = color_img.astype(np.uint8)
        if color_img.shape[-1] == 1 or len(color_img.shape) == 2:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2RGB)
        return color_img

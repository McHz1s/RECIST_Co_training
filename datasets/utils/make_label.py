from functools import partial

import cv2
import numpy as np

from datasets.utils.utils import dispacth_recist_points, recist2extreme, euclidean_distance
from utils.geometry import intersection, recist2det
from utils.mask_operator import post_process_from_diamond, mask_union


class MaskMaker(object):
    def __init__(self, kps_list, mask_shape, ori_img=None, post_process=False, post_th=120):
        self.ori_img = ori_img
        self.post_th = post_th
        self.recist_pts = kps_list
        self.cross_pts_list = []
        self.cross = True
        self.is_post_process = post_process
        for kps in kps_list:
            cross, _ = dispacth_recist_points(kps)
            if cross is None:
                self.cross = False
                cross = recist2extreme(kps)
            self.cross_pts_list.append(cross)
            if self.cross_pts_list[0] is None:
                self.cross_pts_list = []
        self.mask_shape = mask_shape
        self.geometry_pool = ['diamond', 'circle', 'recist_channel', 'erode_diamond', 'erode_circle', 'recist_bbox']

    def recist2geometry(self, shape, mask_name_list):
        mask_dict = {geometry_name: np.zeros(shape, dtype=np.uint8) for geometry_name in mask_name_list}
        for mask_name in mask_name_list:
            for recist_pts, cross_pts in zip(self.recist_pts, self.cross_pts_list):
                if mask_name in self.geometry_pool:
                    self.__getattribute__(mask_name)(mask_dict[mask_name], recist_pts, cross_pts)
                else:
                    temp_mask = self.__getattribute__(mask_name)(recist_pts, cross_pts)
                    if self.is_post_process:
                        temp_mask = post_process_from_diamond(mask_dict['diamond'], temp_mask, self.ori_img,
                                                              self.post_th)
                    mask_dict[mask_name] = mask_union([mask_dict[mask_name], temp_mask])
        return mask_dict

    @staticmethod
    def diamond(mask, recist_pts, cross_pts):
        cv2.fillPoly(mask[0], [cross_pts.astype(np.int)], 1)

    @staticmethod
    def erode_diamond(mask, recist_pts, cross_pts):
        cv2.fillPoly(mask[0], [cross_pts.astype(np.int)], 1)
        mask[0] = cv2.erode(mask[0], kernel=(3, 3), iterations=3)

    @staticmethod
    def recist_channel(mask, recist_pts, cross_pts):
        recist_pts= recist_pts.astype(np.int)
        recist_tuple = tuple(map(tuple, recist_pts))
        cv2.line(mask[0], recist_tuple[0], recist_tuple[1], 1, 2)
        cv2.line(mask[0], recist_tuple[2], recist_tuple[3], 1, 2)

    @staticmethod
    def circle(mask, recist_pts, cross_pts):
        (x, y), radius = cv2.minEnclosingCircle(recist_pts.astype(np.int))
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(mask[0], center, radius, 1, -1)

    @staticmethod
    def erode_circle(mask, recist_pts, cross_pts):
        (x, y), radius = cv2.minEnclosingCircle(recist_pts.astype(np.int))
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(mask[0], center, radius, 1, -1)
        mask[0] = cv2.erode(mask[0], kernel=(3, 3), iterations=3)

    def recist_bbox(self, mask, recist_pts, cross_pts):
        bboxes = recist2det(recist_pts, 5, *mask.shape[1:3])
        for bbox in bboxes:
            bbox = bbox.astype(np.int)
            mask[0, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1

    def recist_simple_uncertain(self, recist_pts, cross_pts):
        d_l1l2 = euclidean_distance(recist_pts[0], recist_pts[1])
        mask_cl1, mask_cl2 = list(map(partial(np.zeros, dtype=np.uint8), [self.mask_shape] * 2))
        cv2.circle(mask_cl1, tuple(recist_pts[0].astype(np.int)), int(d_l1l2), 1, -1)
        cv2.circle(mask_cl2, tuple(recist_pts[1].astype(np.int)), int(d_l1l2), 1, -1)
        u = mask_cl1 * mask_cl2
        u = np.expand_dims(u, axis=0)
        return u

    def recist_uncertain(self, recist_pts, cross_pts):
        d_l1l2 = euclidean_distance(recist_pts[0], recist_pts[1])
        mask_cl1, mask_cl2, mask_rec = list(map(partial(np.zeros, dtype=np.uint8), [self.mask_shape] * 3))
        cv2.circle(mask_cl1, tuple(recist_pts[0].astype(np.int)), int(d_l1l2), 1, -1)
        cv2.circle(mask_cl2, tuple(recist_pts[1].astype(np.int)), int(d_l1l2), 1, -1)
        intct_x, intct_y = intersection(recist_pts[0], recist_pts[1], recist_pts[2], recist_pts[3])
        if intct_x is None:
            u = np.expand_dims(np.zeros_like(mask_cl1), axis=0)
            self.circle(u, recist_pts, cross_pts)
            return u
        delta_x = [recist_pts[2][0] - intct_x, recist_pts[3][0] - intct_x]
        delta_y = [recist_pts[2][1] - intct_y, recist_pts[3][1] - intct_y]
        r_c1_1 = recist_pts[0] + np.array([delta_x[0], delta_y[0]])
        r_c1_2 = recist_pts[0] + np.array([delta_x[1], delta_y[1]])
        r_c2_1 = recist_pts[1] + np.array([delta_x[0], delta_y[0]])
        r_c2_2 = recist_pts[1] + np.array([delta_x[1], delta_y[1]])
        rec_pt_array = np.array([[r_c1_1, r_c1_2, r_c2_2, r_c2_1]]).astype(np.int)
        cv2.fillPoly(mask_rec, [rec_pt_array], 1)
        u = mask_cl1 * mask_cl2 * mask_rec
        u = np.expand_dims(u, axis=0)
        return u

    @staticmethod
    def make_shield(f, u):
        one = np.ones_like(f)
        two = 2 * one
        shield = np.where(f == u, f, two)
        return shield

import cv2
import numpy as np
import math
from skimage import measure
from pycocotools.coco import COCO
import random

from utils.geometry import find_recist_in_contour, close_contour, get_cross_point


def normalize(img):
    img = img.astype('float32')
    img_min = np.min(img)
    img_max = np.max(img)
    img = (img - img_min) / (img_max - img_min)
    if len(img.shape) == 1:
        img = np.expand_dims(img, axis=0)
    return img


def window_level_normalize(img, level, window):
    from utils.CT_preprocess import window_level_normalize
    img = window_level_normalize(img, level, window)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    return img


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[int(y) - top:int(y) + bottom, int(x) - left:int(x) + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


# ##############################
# line sup
# ##############################
def xgety(x, k, b):
    return int(round(k * x + b))


def ygetx(y, k, b):
    return int(round((y - b) / k))


def validate(pt, height, width):
    return 0 <= pt[0] <= height and 0 <= pt[1] <= width


def get_extend_point(pt1, pt2, height=511, width=511):
    y1, x1, y2, x2 = pt1[1], pt1[0], pt2[1], pt2[0]
    if x1 == x2:
        return np.array([0, x1]), np.array([height, x2])
    k = (y1 - y2) / (x1 - x2)
    b = y1 - k * x1
    x_border1, x_border2 = 0, width
    y_border1, y_border2 = 0, height
    intserctino = [[xgety(x_border1, k, b), x_border1], [xgety(x_border2, k, b), x_border2],
                   [y_border1, ygetx(y_border1, k, b)], [y_border2, ygetx(y_border2, k, b)]]
    re = []
    for i in intserctino:
        if validate(i, height, width):
            re.append(i)
    assert len(re) == 2
    return re


# ####################################
# circle sup
# ####################################
def diameter(pt1, pt2):
    return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2


def euclidean_distance(pt1, pt2):
    return math.sqrt((1 * (pt1[0] - pt2[0])) ** 2 + (1 * (pt1[1] - pt2[1])) ** 2)


def recist_euclidean_distance(pt1, pt2, spacing):
    return math.sqrt((spacing[0] * (pt1[0] - pt2[0])) ** 2 + (spacing[1] * (pt1[1] - pt2[1])) ** 2)
    # return math.sqrt((1 * (pt1[0] - pt2[0])) ** 2 + (1 * (pt1[1] - pt2[1])) ** 2)


def judge_long_diameter_pts(lr_pts, tb_pts):
    long_pts, short_pts = lr_pts.copy(), tb_pts.copy()
    if diameter(lr_pts[0], lr_pts[1]) >= diameter(tb_pts[0], tb_pts[1]):
        long_pts, short_pts = short_pts.copy(), long_pts.copy()
    return long_pts, short_pts


def get_radius_center(pts):
    return round(math.sqrt(diameter(pts[0], pts[1])) / 2.), \
           (round((pts[0][0] + pts[1][0]) / 2), round((pts[0][1] + pts[1][1]) / 2))


def bbox2mask(bbox, pts, img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    iterCount = 10
    cv2.grabCut(img, mask, bbox, bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_RECT)
    if pts is None:
        result = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return result
    f_mask = np.zeros_like(img)
    f_mask[:] = cv2.GC_PR_BGD
    cv2.fillConvexPoly(f_mask, pts.astype(np.int), (cv2.GC_FGD, cv2.GC_FGD, cv2.GC_FGD))
    f_mask = f_mask[:, :, 0]
    mask[f_mask == 1] = 1
    cv2.grabCut(img, mask, bbox, bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_MASK)
    result = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result[result != 0] = 1
    return result


def raw_img2color_img(raw_img):
    n_img = window_level_normalize(raw_img, level=200, window=400)
    n_img_255 = 255 * n_img
    n_img_255 = n_img_255.astype(np.uint8)
    c_img = cv2.cvtColor(n_img_255, cv2.COLOR_GRAY2RGB)
    return c_img


def get_poly_center(poly):
    rect = cv2.minAreaRect(poly)
    return rect[0]


def largest_connect_component(img):
    labeled_img, num = measure.label(img, background=0, return_num=True, connectivity=1)
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        num = np.sum(labeled_img == i)
        if num > max_num:
            max_num = num
            max_label = i
    mcr = (labeled_img == max_label)
    return mcr


def get_recist_from_crop_mask(crop_mask):
    if np.max(crop_mask) == 0:
        return None
    mcr = largest_connect_component(crop_mask)
    padded_binary_mask = np.pad(mcr, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    contour = contours[0]
    contour = close_contour(contour)
    contour = measure.approximate_polygon(contour, 2)
    contour = np.flip(contour, axis=1)
    long_diameter, short_diameter, long_diameter_len, short_diameter_len = find_recist_in_contour(contour)
    recist = np.vstack((contour[long_diameter], contour[short_diameter]))
    recist = recist.reshape((-1))
    cross_extreme_point = get_cross_point(recist)
    return cross_extreme_point


def crop_recist_from_roi(pts, roi):
    pts = pts.copy()
    pts[:, 0] -= roi[0]
    pts[:, 1] -= roi[1]
    return pts


def rescale_pts(pts, cur_height, cur_width, out_height, out_width):
    pts = pts.copy()
    pts[:, 0], pts[:, 1] = out_width * pts[:, 0] / cur_width, out_height * pts[:, 1] / cur_height
    return pts


def gen_gt_instance_mask(gt_pts_list, gt_bbox_list, color_img):
    gt_p_mask_list = []
    for bbox, pts in zip(gt_bbox_list, gt_pts_list):
        gt_p_mask_list.append(bbox2mask(bbox, np.array(pts).astype(np.int), color_img))
    return gt_p_mask_list


def prepare_gt_dict(coco_file_path):
    gt_dict = {}
    coco = COCO(coco_file_path)
    for item in coco.imgs:
        img_id = item
        img_name = coco.imgs[item]['file_name']
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if img_name not in gt_dict:
                gt_dict[img_name] = {
                    'extreme_points_list': [ann['extreme_points']],
                    # 'mask_path_list': [ann['mask_path']],
                }
            else:
                # gt_dict[batch_img_name]['mask_path_list'].append(ann['mask_path'])
                gt_dict[img_name]['extreme_points_list'].append(ann['extreme_points'])
    return gt_dict


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return np.array([-1, -1])

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])


# def dispacth_recist_points(recist_points):
#     inter = line_intersection(recist_points[:2], recist_points[2:])
#     if any(inter <= 0):
#         return None, None
#     int_x, int_y = inter[0], inter[1]
#     region_map = {(1, 1): 0, (1, -1): 1, (-1, -1): 2, (-1, 1): 3}
#     data_dict = np.zeros((4, 2))
#     judge = lambda x1, y1: (1 if x1 > int_x else -1, 1 if y1 >= int_y else -1)
#     long1_r = region_map[judge(recist_points[0, 0], recist_points[0, 1])]
#     if long1_r == 0 or long1_r == 2:
#         data_dict[long1_r], data_dict[2 - long1_r] = recist_points[0], recist_points[1]
#         short1_r = 1 if recist_points[2, 0] <= recist_points[3, 0] else 3
#         data_dict[short1_r], data_dict[4 - short1_r] = recist_points[2], recist_points[3]
#     else:
#         data_dict[long1_r], data_dict[4 - long1_r] = recist_points[0], recist_points[1]
#         short1_r = 2 if recist_points[2, 0] <= recist_points[3, 0] else 0
#         data_dict[short1_r], data_dict[2 - short1_r] = recist_points[2], recist_points[3]
#     return data_dict, inter

def dispacth_recist_points(recist_points):
    inter = line_intersection(recist_points[:2], recist_points[2:])
    if any(inter <= 0):
        return None, None
    mid_x, mid_y = inter[0], inter[1]
    region_map = {(-1, -1): 0, (1, -1): 1, (1, 1): 2, (-1, 1): 3}
    re = np.zeros((4, 2))
    judge = lambda x1, y1: (1 if y1 - x1 + mid_x - mid_y >= 0 else -1, 1 if y1 + x1 - mid_x - mid_y >= 0 else -1)
    long1_r = region_map[judge(recist_points[0, 0], recist_points[0, 1])]
    if long1_r == 0 or long1_r == 2:
        re[long1_r], re[2 - long1_r] = recist_points[0], recist_points[1]
        short1_r = 1 if recist_points[2, 0] <= recist_points[3, 0] else 3
        re[short1_r], re[4 - short1_r] = recist_points[2], recist_points[3]
    else:
        re[long1_r], re[4 - long1_r] = recist_points[0], recist_points[1]
        short1_r = 2 if recist_points[2, 0] <= recist_points[3, 0] else 0
        re[short1_r], re[2 - short1_r] = recist_points[2], recist_points[3]
    return re, inter


def left_right(in_recist_points):
    recist_points = in_recist_points.copy()
    if recist_points[0, 0] > recist_points[1, 0]:
        recist_points[0:2] = np.flip(recist_points[0:2], axis=0)
    if recist_points[2, 0] > recist_points[3, 0]:
        recist_points[2:4] = np.flip(recist_points[2:4], axis=0)
    return recist_points


def recist2extreme(pt):
    # get extreme points
    x = pt[:, 0]
    y = pt[:, 1]
    top_idx = np.argmin(y)
    xt, yt = x[top_idx], y[top_idx]
    bottom_idx = np.argmax(y)
    xb, yb = x[bottom_idx], y[bottom_idx]
    left_idx = np.argmin(x)
    xl, yl = x[left_idx], y[left_idx]
    right_idx = np.argmax(x)
    xr, yr = x[right_idx], y[right_idx]
    return np.array([[xt, yt], [xl, yl], [xb, yb], [xr, yr]])


def percent_noise(percent=0.2):
    return random.random() * percent


def get_delta(point1, point2):
    x_dis, y_dis = abs(point2[0] - point1[0]), abs(point2[1] - point1[1])
    x1_delta, x2_delta = percent_noise() * x_dis, percent_noise() * x_dis
    if point2[0] == point1[0]:
        return x1_delta, x2_delta, percent_noise() * y_dis, percent_noise() * y_dis
    y1_delta, y2_delta = x1_delta * y_dis / x_dis, x2_delta * y_dis / x_dis
    return x1_delta, x2_delta, y1_delta, y2_delta


def noise_recist(recist_pts, height, width, point=[0, 2]):
    recist_pts = recist_pts.copy()
    symbol_map = lambda x: 1 if x <= 0.5 else -1
    x1_delta, x2_delta, y1_delta, y2_delta = get_delta(recist_pts[point[0]], recist_pts[point[1]])
    recist_pts[point[0]] += symbol_map(random.random()) * np.array([x1_delta, y1_delta])
    recist_pts[point[1]] += symbol_map(random.random()) * np.array([x1_delta, y1_delta])
    recist_pts[:, 0] = np.clip(recist_pts[:, 0], 1, width - 2)
    recist_pts[:, 1] = np.clip(recist_pts[:, 1], 1, height - 2)
    return recist_pts

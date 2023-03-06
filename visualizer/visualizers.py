import io

import cv2
import numpy as np
from PIL import Image

COLOR_POOL = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 255, 0)]


def gen_colormap(img, cl=(255, 0, 0), s=4):
    img[img < 0] = 0
    h, w = img.shape
    color_map = np.zeros((h * s, w * s, 3), dtype=np.uint8)
    resized = cv2.resize(img, (w * s, h * s)).reshape(h * s, w * s, 1)
    color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map


def vis_mask(img, mask, color=(255, 0, 0), beta=0.2):
    show_img = vis_img(img, None)
    color_mask = np.zeros_like(img, dtype=img.dtype)
    color_mask[mask == 1] = np.array(color)
    show = cv2.addWeighted(show_img, 1, color_mask, beta, 0)
    return show


def vis_many_mask(img, mask_list, color_list=None, beta=0.2):
    show_img = vis_img(img, None)
    if not isinstance(mask_list, list):
        mask_list = [mask_list]
    if color_list is None:
        color_list = COLOR_POOL
    for m_i, mask in enumerate(mask_list):
        show_img = vis_mask(show_img, mask, color_list[m_i], beta)
    return show_img


def vis_mask_outline(img, mask, thickness=1):
    if len(img.shape) == 2 or img.shape[2] == 1:
        show = vis_img(img, None)
    else:
        show = img.copy()
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        show = cv2.drawContours(show, [c], 0, (0, 0, 255), thickness)
    return show


def add_blend_img(back, fore, trans=0.5):
    # fore = 255 - fore
    if fore.shape[0] != back.shape[0] or fore.shape[1] != back.shape[1]:
        fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
        fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    img = (back * (1. - trans) + fore * trans)
    img[img > 255] = 255
    img = img.astype(np.uint8)
    return img


def vis_img(img, use_less=None):
    img = img.copy()
    if len(img.shape) == 2 or img.shape[2] == 1:
        if np.max(img) <= 1:
            img *= 255
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img = img.astype(np.uint8)
    return img


def vis_img_hm(img, hmp):
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB)
    hmp = hmp.squeeze()
    hmp = gen_colormap(hmp)
    img_hmp = add_blend_img(img, hmp)
    return img_hmp


def vis_many_hm(img, hmp_sequence):
    img = img.copy()
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB)
    if not isinstance(hmp_sequence, list) and len(hmp_sequence.shape) == 2:
        return vis_img_hm(img, hmp_sequence)
    for hmp in hmp_sequence:
        img = vis_img_hm(img, hmp.squeeze())
    return img


def vis_cimg_hm(cimg, hmp):
    cimg = cimg.copy()
    hmp = hmp.copy()
    hmp = hmp.squeeze()
    hmp = gen_colormap(hmp)
    img_hmp = add_blend_img(cimg, hmp)
    return img_hmp


def vis_cimg_distinct_mask(cimg, maskList, colorList=None):
    show_img = cimg.copy()
    maskList = [mask.copy().squeeze() for mask in maskList]
    # hardcode, only support dual mask
    if colorList is None:
        colorList = COLOR_POOL[:len(maskList)]
    for mask, color in zip(maskList, colorList):
        show_img = vis_mask(show_img, mask, color)
    return show_img


def vis_img_bbox(img, gts, preds):
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.uint8)
    color = (255, 0, 0)  # red
    for bbox in gts:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
    color = (0, 255, 0)  # green
    for pred in preds:
        bbox, score = pred[:4], pred[4]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
    return img


def vis_bbox(img, gts):
    img = img.copy()
    color = (255, 0, 0)  # red
    for bbox in gts:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
    return img


def vis_a_bbox(img, bbox):
    color = (255, 0, 0)  # red
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]) + int(bbox[2]), int(bbox[2]) + int(bbox[1])), color,
                  1)
    return img


def vis_feature(img, feature_map):
    img = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.uint8)
    # normalize feature map
    h, w, _ = img.shape
    feature_map = cv2.resize(feature_map, (w, h)).reshape(h, w, 1)
    norm_img = np.zeros(feature_map.shape)
    norm_img = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    # apply color map
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    heat_img = heat_img.astype(np.uint8)
    # add to batch_img
    img_add = cv2.addWeighted(img, 0.3, heat_img, 0.7, 0)
    img_add = img_add.astype(np.uint8)
    return img_add


def vis_a_poly(img, contours):
    contours = list(map(int, contours[0]))
    contours = np.array(contours)
    contours = np.reshape(contours, (-1, 2))
    cv2.drawContours(img, [contours], 0, (0, 0, 255), 2)
    return img


def vis_a_recist(img, recist):
    img = img.copy()
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    img = img.astype(np.uint8)
    cv2.line(img, (int(recist[0][0]), int(recist[0][1])), (int(recist[2][0]), int(recist[2][1])), (255, 0, 255),
             2)  # top bottom
    cv2.line(img, (int(recist[1][0]), int(recist[1][1])), (int(recist[3][0]), int(recist[3][1])), (0, 255, 0),
             2)  # left right
    return img


def vis_a_real_recist(img, recist_pts, thickness=1):
    img = img.copy()
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    img = img.astype(np.uint8)
    recist_pts = np.array(recist_pts).astype(np.int).reshape((4, 2))
    recist_tuple = tuple(map(tuple, recist_pts))
    cv2.line(img, recist_tuple[0], recist_tuple[1], (255, 0, 255), thickness)
    cv2.line(img, recist_tuple[2], recist_tuple[3], (0, 255, 0), thickness)
    return img


def vis_real_recist(img, recist_sequence, thickness=1):
    img = img.copy()
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    if not isinstance(recist_sequence, list) and len(recist_sequence.shape) == 2:
        return vis_a_real_recist(img, recist_sequence, thickness)
    for recist in recist_sequence:
        img = vis_a_real_recist(img, recist, thickness)
    return img


def vis_a_pre_gt_recist(img, pre_recist, gt_recist):
    img = img.copy()
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    img = img.astype(np.uint8)
    cv2.line(img, (int(pre_recist[0][0]), int(pre_recist[0][1])), (int(pre_recist[2][0]), int(pre_recist[2][1])),
             (0, 255, 0), 2)  # top bottom
    cv2.line(img, (int(pre_recist[1][0]), int(pre_recist[1][1])), (int(pre_recist[3][0]), int(pre_recist[3][1])),
             (0, 255, 0), 2)  # left right4
    cv2.line(img, (int(gt_recist[0][0]), int(gt_recist[0][1])), (int(gt_recist[2][0]), int(gt_recist[2][1])),
             (255, 0, 0), 2)  # top bottom
    cv2.line(img, (int(gt_recist[1][0]), int(gt_recist[1][1])), (int(gt_recist[3][0]), int(gt_recist[3][1])),
             (255, 0, 0), 2)  # left right
    return img


def plt_show(x, title=None):
    import matplotlib.pyplot as plt
    plt.imshow(x)
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()


def vis_img_seg_bbox(img, bboxes, segmap, color=(0, 255, 0)):
    img = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB)
    segmap = gen_colormap(segmap, cl=color, s=1)

    img = segmap * 0.5 + img
    img[img > 255] = 255
    img = img.astype(np.uint8)

    for bbox in bboxes:
        bbox = bbox[:4]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
    return img


def pltAxes2npArray(plt_ax):
    canvas = plt_ax.figure.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    arr = Image.open(buffer)
    arr = np.asarray(arr)
    return arr

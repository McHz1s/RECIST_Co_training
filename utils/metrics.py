##########################################
# metric
##########################################
import scipy
import numpy as np


def cal_dice(gt_seg, pred_seg, threshold=0.5):
    if pred_seg is None:
        pred_seg = np.zeros_like(gt_seg)
    pred_seg[pred_seg > threshold] = 1
    pred_seg[pred_seg <= threshold] = 0
    top = 2 * np.sum(gt_seg * pred_seg)
    bottom = np.sum(gt_seg) + np.sum(pred_seg)
    bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
    dicem = top / bottom
    return dicem


def margin_of_error_at_confidence_level(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

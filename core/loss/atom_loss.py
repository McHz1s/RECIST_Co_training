import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .loss import ATOMLOSS

__all__ = ['focal_loss', 'neg_loss', 'reg_loss',
           'bce_loss', 'log_bce_loss', 'dice_loss', 'log_dice_loss',
           'weights_balance_bce_loss', 'weights_balance_log_bce_loss']

# # focal_loss ################################
from ..network.utils import sigmoid


def _focal_loss(pred, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.float().sum()
    neg_loss = neg_loss.float().sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


@ATOMLOSS
def focal_loss():
    return _focal_loss


# # dice_loss ################################
def diceCoeff(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1).float()
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N


class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeff(y_pr, y_gt)


class SoftDiceLossWithLogits(SoftDiceLoss):
    __name__ = 'dice_loss'

    def __init__(self):
        super(SoftDiceLossWithLogits, self).__init__()

    def forward(self, y_pr, y_gt):
        y_pr = y_pr.float()
        y_pr = torch.sigmoid(y_pr)
        return 1 - diceCoeff(y_pr, y_gt)


###############################################
# detection
###############################################

@ATOMLOSS
def neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


@ATOMLOSS
def reg_loss(output, mask, ind, gt_regr):
    """
    Regression loss for an output tensor
        Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """
    regr = _transpose_and_gather_feat(output, ind)

    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)

    return regr_loss


class LogisticMSELoss(nn.Module):
    def __init__(self):
        super(LogisticMSELoss, self).__init__()

    def forward(self, pred, target, reduction='mean'):
        pred = pred.float()
        pred = sigmoid(pred)
        return F.mse_loss(pred, target, reduction=reduction)


class TargetLogisticMSELoss(LogisticMSELoss):
    def __init__(self):
        super(LogisticMSELoss, self).__init__()

    def forward(self, pred, target, reduction='mean'):
        target = target.float()
        target = sigmoid(target)
        return super(TargetLogisticMSELoss, self).forward(pred, target,
                                                          reduction=reduction)


class PartialBCELogitsLoss(nn.Module):
    def __init__(self):
        super(PartialBCELogitsLoss, self).__init__()

    def calculate_category_loss(self, prob_map, p_mask, customized_weights=None):
        b, c, h, w = p_mask.size()
        batch_weight = 1 / p_mask.sum(dim=[-1, -2])
        batch_weight = batch_weight.view([b, c, 1, 1]).expand([b, c, h, w])
        batch_weight = batch_weight.clone()
        batch_weight[p_mask != 1] = 0
        if customized_weights is not None:
            batch_weight = torch.where(customized_weights != 0, customized_weights, batch_weight)
        par_loss = F.binary_cross_entropy_with_logits(prob_map, torch.ones_like(p_mask),
                                                      weight=batch_weight, reduction='sum')
        par_loss /= b
        return par_loss

    def forward(self, prob_map, f_mask=None, b_mask=None, customized_f_weights=None, customized_b_weights=None):
        par_f_loss = 0 if f_mask is None else self.calculate_category_loss(prob_map, f_mask, customized_f_weights)
        par_b_loss = 0 if b_mask is None else self.calculate_category_loss(1 - prob_map, b_mask, customized_b_weights)
        return par_f_loss, par_b_loss


class PartialDiceLogitsLoss(nn.Module):
    def __init__(self):
        super(PartialDiceLogitsLoss, self).__init__()

    def forward(self, prob_map, f_mask, b_mask):
        b, c, h, w = f_mask.size()
        batch_weight = 1 / f_mask.sum(dim=[-1, -2])
        batch_weight = batch_weight.view([b, c, 1, 1]).expand([b, c, h, w])
        batch_weight = batch_weight.clone()
        batch_weight[f_mask != 1] = 0
        par_f_loss = F.binary_cross_entropy_with_logits(prob_map, torch.ones_like(f_mask),
                                                        weight=batch_weight, reduction='sum')
        batch_weight = 1 / b_mask.sum(dim=[-1, -2])
        batch_weight = batch_weight.view([b, c, 1, 1]).expand([b, c, h, w])
        batch_weight = batch_weight.clone()
        batch_weight[b_mask != 1] = 0
        par_b_loss = F.binary_cross_entropy_with_logits(prob_map, torch.zeros_like(b_mask),
                                                        weight=batch_weight, reduction='sum')
        return par_f_loss, par_b_loss


class WeightBalanceBCELoss(nn.Module):
    def __init__(self):
        super(WeightBalanceBCELoss, self).__init__()

    def get_weights(self, gt):
        b, c, h, w = gt.size()
        total = h * w
        pos = torch.eq(gt, 1).float()
        neg = torch.eq(gt, 0).float()
        batch_num_neg = torch.clamp(torch.sum(neg, dim=[-1, -2]), max=int(h * w * 0.2))
        batch_num_pos = total - batch_num_neg
        batch_alpha_pos = 0.5 * total / batch_num_pos
        batch_pos_weight = pos * batch_alpha_pos.view([b, c, 1, 1]).expand([b, c, h, w])
        batch_alpha_neg = 0.5 * total / batch_num_neg
        batch_neg_weight = neg * batch_alpha_neg.view([b, c, 1, 1]).expand([b, c, h, w])
        weights = (batch_pos_weight + batch_neg_weight)
        return weights

    def forward(self, pred, gt):
        weights = self.get_weights(gt)
        b, _, h, w = gt.size()
        return F.binary_cross_entropy(pred, gt, weights, reduction='mean')


class WeightBalanceBCELossWithLogits(WeightBalanceBCELoss):
    def __init__(self):
        super(WeightBalanceBCELossWithLogits, self).__init__()

    def forward(self, pred, gt):
        weights = self.get_weights(gt)
        return F.binary_cross_entropy_with_logits(pred, gt, weights, reduction='mean')


class Grad(nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        pred_shape = y_pred.shape
        if len(pred_shape) == 4:
            y_pred = torch.unsqueeze(y_pred, dim=2)
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        dy_mean = 0 if pred_shape[2] == 1 or len(pred_shape) == 4 else torch.mean(dy)
        d = torch.mean(dx) + torch.mean(dz) + dy_mean
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class NCCLoss(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCCLoss, self).__init__()
        self.win = win

    def forward(self, y_pred, y_true):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


# ------------------------------------------------------------------------------
@ATOMLOSS
def bce_loss():
    return nn.BCELoss()


@ATOMLOSS
def weights_balance_bce_loss():
    return WeightBalanceBCELoss()


@ATOMLOSS
def weights_balance_log_bce_loss():
    return WeightBalanceBCELossWithLogits()


@ATOMLOSS
def log_bce_loss():
    return nn.BCEWithLogitsLoss()


@ATOMLOSS
def dice_loss():
    return SoftDiceLoss()


@ATOMLOSS
def log_dice_loss():
    return SoftDiceLossWithLogits()


@ATOMLOSS
def mse_loss():
    return nn.MSELoss()


@ATOMLOSS
def log_mse_loss():
    return LogisticMSELoss()


@ATOMLOSS
def target_log_mse_loss():
    return TargetLogisticMSELoss()


@ATOMLOSS
def ncc_loss():
    return NCCLoss()


@ATOMLOSS
def grad_loss():
    return Grad()


@ATOMLOSS
def partial_bce_logits_loss():
    return PartialBCELogitsLoss()

import re

import torch
import torch.nn as nn

from core.network.utils import sigmoid
from .. import model_register

vgg19_block_out_channel = [64, 128, 256, 512, 512]
resnet50_block_out_channel = [64, 256, 512, 1024, 2048]
resnet18_block_out_channel = [64, 64, 128, 256, 512]


@model_register
class HNN(nn.Module):
    def __init__(self, cfg):
        super(HNN, self).__init__()
        self.cfg = cfg.to_dict()
        self.cfg.pop('backbone')
        backbone_name = cfg.backbone
        self.backbone_name = backbone_name
        self.backbone = globals()[backbone_name](**self.cfg)
        backbone_type = re.sub(r'\d+', '', backbone_name)
        aggre_conv_list = [nn.Conv2d(out_channel, 1, kernel_size=1, stride=1, padding=0)
                           for out_channel in globals()[f'{self.backbone_name}_block_out_channel']]
        self.agreg_conv_list = nn.ModuleList(aggre_conv_list)
        upsample_list = [nn.Upsample(scale_factor=2**i, mode='bilinear') for i in range(1, 6)]
        self.upsample_list = nn.ModuleList(upsample_list)
        self.netCombine = torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        backbone_feature_list = self.backbone(img)
        # output_dict = {'backbone_feature': backbone_feature_list}
        aggre_feature_list = []
        for aggre_conv, upsample, feature in zip(self.agreg_conv_list, self.upsample_list, backbone_feature_list):
            aggre_feature_list.append(upsample(aggre_conv(feature)))
        aggre = torch.cat(aggre_feature_list, dim=1)
        comb = self.netCombine(aggre)
        prob_map = sigmoid(comb)
        output_dict = {'prob_map': prob_map}
        return output_dict

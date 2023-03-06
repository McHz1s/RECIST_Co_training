import cv2
import numpy as np

from core.network.post_process import POSTPROCESSOR
from core.network.post_process.basic_post_process import BasicPostProcess
from core.network.utils import sigmoid
from utils.mask_operator import prob2mask
from utils.tensor_operator import petrificus_totalus


@POSTPROCESSOR.register
class BinarySemanticSegmentationPostProcess(BasicPostProcess):
    def __init__(self, post_process_cfg):
        super(BinarySemanticSegmentationPostProcess, self).__init__(post_process_cfg)

    def is_sigmoid(self, inputs):
        inputs = inputs.clone().detach()
        if self.cfg.get('sigmoid', False):
            inputs = sigmoid(inputs)
        return inputs

    def forward(self, output_dict, img_name, batch_reverse_transform):
        prob_map = petrificus_totalus(self.is_sigmoid(output_dict['prob_map']).float())
        post_out_dict = self.reverse_and_convert_to_mask(prob_map, img_name, batch_reverse_transform)
        return post_out_dict

    def to_mask(self, probability_map, img_name, batch_reverse_transform):
        probability_map = petrificus_totalus(probability_map)
        probability_map = np.transpose(probability_map, [0, 2, 3, 1])
        mask = prob2mask(probability_map, self.cfg.mask_threshold).astype(np.uint8)
        mask_dict = {n: r(m).astype(np.uint8) for n, r, m in zip(img_name, batch_reverse_transform, mask)}
        return mask_dict

    def reverse_and_convert_to_mask(self, probability_map, img_name, batch_reverse_transform):
        probability_map = petrificus_totalus(probability_map)
        probability_map = np.transpose(probability_map, [0, 2, 3, 1])
        post_dict = dict()
        for n, r, p in zip(img_name, batch_reverse_transform, probability_map):
            rp = r(p, interpolation=cv2.INTER_LINEAR)
            rm = r(prob2mask(p, self.cfg.mask_threshold), interpolation=cv2.INTER_NEAREST)
            post_dict[n] = {'pred_mask': rm, 'prob_map': rp}
        return post_dict

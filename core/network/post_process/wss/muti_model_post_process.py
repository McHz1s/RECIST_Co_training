from core.network.post_process import POSTPROCESSOR
from utils.tensor_operator import petrificus_totalus
from .binary_segmentation_post_process import BinarySemanticSegmentationPostProcess


@POSTPROCESSOR.register
class MultiModelBinarySemanticSegmentationPostProcess(BinarySemanticSegmentationPostProcess):
    def __init__(self, post_process_cfg):
        super(MultiModelBinarySemanticSegmentationPostProcess, self).__init__(post_process_cfg)

    def forward(self, mul_model_output_dict, img_name, batch_reverse_transform):
        post_out_dict = {name: {} for name in img_name}
        mul_model_prob_map = [weight * petrificus_totalus(self.is_sigmoid(opt_dict['prob_map']))
                              for weight, opt_dict in zip(self.cfg.model_weight_list, mul_model_output_dict.values())]
        vote = sum(mul_model_prob_map)
        mask_dict = self.to_mask(vote, img_name, batch_reverse_transform)
        for key in post_out_dict:
            post_out_dict[key].update({'pred_mask': mask_dict[key]})
        return post_out_dict

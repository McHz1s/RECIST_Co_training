from core.network import network_register
from core.network.model import get_model
from core.network.post_process import get_post_process
from core.network.test_time_aug import TTA_REGISTER
from core.network.wrapper.basic_wrapper import BasicWrapper


@network_register
class NormalSeg2D(BasicWrapper):
    def __init__(self, cfg_network_part):
        super(NormalSeg2D, self).__init__(cfg_network_part)
        # Import seg net
        self.model_name_list = []
        self.test_aug = TTA_REGISTER.build_with_cfg(self.cfg.get('tta', None))
        self.tta_effect = False
        self.model_name = self.cfg.model_name
        self.logger(f'Init Model: {self.model_name}')
        self.__setattr__(self.model_name, get_model(self.model_name, self.cfg[self.model_name]))
        self.logger(f'Init PostProcessor')
        self.post_process = get_post_process(self.cfg.post_process)
        self.post_process_effect = True

    def segmentation_model_forward(self, img):
        output = self.__getattr__(self.model_name)(img)
        return output

    def forward(self, inp):
        img = inp['img']
        output_dict = self.segmentation_model_forward(img)
        output_dict = self.is_tta_forward(img, output_dict)
        if self.post_process_effect:
            output_dict.update(self.post_process(output_dict, inp['img_name'], inp['reverse_transform']))
        return output_dict

    def is_tta_forward(self, img, output_dict):
        if not self.tta_effect or self.test_aug is None or not self.post_process_effect:
            return output_dict
        aug_out_list = [output_dict]
        for trans_name, (trans, r_trans) in self.test_aug.yield_aug():
            aug_img = trans(img)
            trans_output_dict = self.segmentation_model_forward(aug_img)
            rtrans_output_dict = r_trans(trans_output_dict)
            aug_out_list.append(rtrans_output_dict)
        output_dict = self.test_aug.mean(aug_out_list)
        return output_dict

    def MMCVImageNetConvert(self, pretrained_state_dict):
        p_state_dict = pretrained_state_dict['state_dict']
        convert_dict = {f'net.{key}': value for key, value in p_state_dict.items()}
        return convert_dict

    def HRNetImageNetConvert(self, pretrained_state_dict):
        convert_dict = {f'net.{key}': value for key, value in pretrained_state_dict.items()}
        return convert_dict

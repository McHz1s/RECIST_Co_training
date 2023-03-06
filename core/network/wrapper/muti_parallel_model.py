import re

from core.network import network_register
from core.network.model import get_model
from core.network.post_process import get_post_process
from core.network.test_time_aug import TTA_REGISTER
from core.network.wrapper.basic_wrapper import BasicWrapper


@network_register
class MutiParallelModel(BasicWrapper):
    def __init__(self, cfg_network_part):
        super(MutiParallelModel, self).__init__(cfg_network_part)
        # Import seg net
        self.model_name_list = []
        self.test_aug = TTA_REGISTER.build_with_cfg(self.cfg.get('tta', None))
        self.tta_effect = False
        for i, model_name in enumerate(self.cfg.name_list):
            self.model_name_list.append(f'{model_name}_{i}')
            self.logger(f'Init Model: {self.model_name_list[-1]}')
            self.__setattr__(self.model_name_list[-1], get_model(model_name, self.cfg[model_name]))
        self.logger(f'Init PostProcessor')
        self.post_process = get_post_process(self.cfg.post_process)
        self.post_process_effect = True

    def forward(self, inp):
        img = inp['img']
        output_dict = self.muti_model_forward(img)
        output_dict = self.is_tta_forward(img, output_dict)
        if self.post_process_effect:
            post_inputs = {output_name: output for output_name, output in output_dict.items() if 'out' in output_name}
            output_dict.update(self.post_process(post_inputs, inp['img_name'], inp['reverse_transform']))
        return output_dict

    def is_tta_forward(self, img, output_dict):
        if not self.tta_effect or self.test_aug is None or not self.post_process_effect:
            return output_dict
        aug_out_list = [output_dict]
        for trans_name, (trans, r_trans) in self.test_aug.yield_aug():
            aug_img = trans(img)
            trans_output_dict = self.muti_model_forward(aug_img)
            rtrans_output_dict = r_trans(trans_output_dict)
            aug_out_list.append(rtrans_output_dict)
        output_dict = self.test_aug.mean(aug_out_list)
        return output_dict

    def muti_model_forward(self, img):
        output_dict = {}
        for model_name in self.model_name_list:
            output_dict[f'{model_name}_out'] = self.__getattr__(model_name)(img)
        return output_dict

    def convert_pretrained(self, pretrained_state_dict):
        mul_convert_dict = {}
        for name in self.model_name_list:
            convert_dict = {f'{name}.net.{key}': value for key, value in pretrained_state_dict['state_dict'].items()}
            mul_convert_dict.update(convert_dict)
        return mul_convert_dict

    def MSTConvert(self, pretrained_state_dict):
        mul_convert_dict = {}
        for name in self.model_name_list:
            if 'MSTfamily' not in name:
                continue
            convert_dict = {f'{name}.net.{key}': value for key, value in pretrained_state_dict['state_dict'].items()}
            mul_convert_dict.update(convert_dict)
        return mul_convert_dict

    def muti_model_convert(self, state_dict_list):
        mul_convert_dict = state_dict_list[0]['network']
        for model_num, state_dict in enumerate(state_dict_list[1:]):
            new_state_dict = {}
            state_dict = state_dict['network']
            for weight_name, weight in state_dict.items():
                new_weight_name = re.sub(r'_\d+', f'_{model_num + 1}', weight_name)
                new_state_dict[new_weight_name] = weight
            mul_convert_dict.update(new_state_dict)
        return mul_convert_dict

    def MMCVImageNetConvert(self, pretrained_state_dict):
        mul_convert_dict = {}
        p_state_dict = pretrained_state_dict['state_dict']
        for name in self.model_name_list:
            if 'MSTfamily' not in name:
                continue
            convert_dict = {f'{name}.net.{key}': value for key, value in p_state_dict.items()}
            mul_convert_dict.update(convert_dict)
        return mul_convert_dict

import os

import torch
from torch.cuda.amp import autocast

from core.network import get_network
from core.runner.basic.sheep import Sheep


class Dog(Sheep):
    def __init__(self, *args, **kwargs):
        super(Dog, self).__init__(*args, **kwargs)
        # Setup network
        self.network = None
        self.init_network()
        self.run_model = self.network
        self.set_device_in_out_need_tamplate()
        if self.cfg["run"]["resume"] is not None:
            self.resume()

    @property
    def input_target_dict_name_list(self):
        return ['gpu_input_need_dict',
                'gpu_target_need_dict',
                'cpu_input_need_dict',
                'cpu_target_need_dict',
                'identity_input_need_dict',
                'identity_target_need_dict',
                ]

    def set_device_in_out_need_tamplate(self):
        for need_name in self.input_target_dict_name_list:
            self.__setattr__(need_name, self.cfg.run.get(need_name, {}))

    def init_network(self):
        network_cfg = self.cfg.network
        self.network = get_network(network_cfg).to(self.device)
        self.network.tta_effect = self.cfg.run.get('tta_effect', False)
        if network_cfg.pretrained and self.cfg["run"]["resume"] is None:
            self.network.load_pretrained_params(network_cfg.pretrained_path, network_cfg.strict)
            self.logger(f'Loaded pretrained model from "{network_cfg.pretrained_path}"')
        self.logger('>> Total params: %.2fM' % (sum(p.numel() for p in self.network.parameters()) / 1000000.0))

    def resume(self):
        if 'convert_pretrained' in self.cfg.network:
            self.network.cfg.convert_pretrained = None
        if os.path.isfile(self.cfg["run"]["resume"]):
            checkpoint = torch.load(self.cfg["run"]["resume"], map_location='cpu')
            self.network.load_pretrained_params(None, True, checkpoint)
            self.logger(
                "Loading checkpoint '{}' (epoch {})".format(self.cfg["run"]["resume"], checkpoint["epoch"]))
        else:
            self.logger("No checkpoint found at '{}'".format(self.cfg["run"]["resume"]))

    def make_in_out_target(self, data_dict, need_target=True):
        raise not NotImplementedError

    def FP16_network_forward(self, is_fp16=False, *args, **kwargs):
        if is_fp16:
            with autocast():
                outdict = self.run_model(*args, **kwargs)
        else:
            outdict = self.run_model(*args, **kwargs)
        return outdict

    def get_pred(self, data_dict):
        inp = self.make_in_out_target(data_dict, need_target=False)['input']
        out_dict = self.FP16_network_forward(self.cfg.run.get('FP16', False), inp)
        for_pred_dict = {name: item for name, item in out_dict.items() if not isinstance(name, str) or 'out' not in name}
        return for_pred_dict

    def evaluate(self, pred_dict, dataset_name):
        eval_info = self.__getattribute__(f'{dataset_name}_evaluator').evaluate(pred_dict)
        return eval_info

    def prepare_test(self):
        self.run_model.eval()
        self.network.post_process_effect = True
        super(Dog, self).prepare_test()

    def after_evaluation(self, eval_info):
        self.network.post_process_effect = False
        super(Dog, self).after_evaluation(eval_info)

    def test(self):
        self.network.switch_network_status('testing')
        with torch.no_grad():
            super(Dog, self).test()

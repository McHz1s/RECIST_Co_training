from core.runner.basic.pigs import Pigs
from utils.my_containers import deep_dict_func, deep_dict_get
from utils.tensor_operator import to_device
from utils.utils import INT_MAX


class Runner(Pigs):
    def __init__(self, *args, **kwargs):
        super(Runner, self).__init__(*args, **kwargs)

    def make_in_out_target(self, data_dict, need_target=True):
        need = ['img']
        inp = dict(zip(need, list(map(lambda x: data_dict[x].float().to(self.device), need))))
        if self.runner_status in [self._runner_validing, self._runner_testing]:
            inp['img_name'] = data_dict['img_name']
        if 'reverse_transform' in data_dict:
            inp['reverse_transform'] = data_dict['reverse_transform']
        if not need_target:
            return {'input': inp}
        target = {}
        target_key = ['psuedo_mask_dict', 'shield_mask']
        target.update({key: to_device(value, self.device, 'float')
                       for key, value in data_dict.items() if key in target_key})
        target['img'] = data_dict['img']
        return {'input': inp, 'target': target}

    def before_train_epoch(self, data_type):
        super(Runner, self).before_train_epoch(data_type)
        epoch_delay = deep_dict_get(self.cfg.run.loss.cfg, 'epoch_delay', INT_MAX)
        if epoch_delay < self.epoch:
            deep_dict_func(self.cfg.run.loss.cfg, lambda x: INT_MAX, 'cons_delay')
            deep_dict_func(self.cfg.run.loss.cfg, lambda x: INT_MAX, 'epoch_delay')

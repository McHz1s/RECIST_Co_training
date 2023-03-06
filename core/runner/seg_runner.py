from core.runner.basic.pigs import Pigs


class Runner(Pigs):
    def __init__(self, *args, **kwargs):
        super(Runner, self).__init__(*args, **kwargs)

    def make_in_out_target(self, data_dict, need_target=True):
        super_made = super(Runner, self).make_in_out_target(data_dict, need_target)
        inp, target = super_made['input'], super_made['target']
        inp['img_name'] = data_dict.get('img_name', 'placeholder')
        if 'reverse_transform' in data_dict:
            inp['reverse_transform'] = data_dict['reverse_transform']
        if not need_target:
            return {'input': inp}
        return {'input': inp, 'target': target}

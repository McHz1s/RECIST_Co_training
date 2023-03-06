import re

import tqdm

from utils.loggers import LoggerManager
from utils.metrics import cal_dice
from utils.tensor_operator import petrificus_totalus


class Evaluator(object):
    def __init__(self, gt_data_loader, writer, cfg, gt_need, csv_logger=None):
        self.gt_data_loader = gt_data_loader
        self.logger = LoggerManager().get_logger('Evaluator')
        self.writer = writer
        self.csv_logger = csv_logger
        self.cfg = cfg
        self.primary_key = self.cfg.pop('primary_key', 'img_name')
        self.gt_need = gt_need
        self.gt_dict = self.prepare_gt()
        self.pre_dict = None

    def auto_evaluation(self, pred_dict, gt_dict):
        eval_info = {}
        for pred_name, pred_value in pred_dict.items():
            if 'mask' in pred_name:
                eval_info.update({'Dice': cal_dice(gt_dict['gt_mask'], pred_value)})
        return eval_info

    def prepare_gt(self):
        gt_anns = {}
        for batch_ind, data_dict in tqdm.tqdm(enumerate(self.gt_data_loader), 'Init Evaluator...'):
            gt_anns[data_dict[self.primary_key][0]] = {k: petrificus_totalus(v)[0]
                                                       for k, v in data_dict.items() if k in self.gt_need}
        return gt_anns

    def evaluate(self, pre_dict):
        result_dict = {}
        self.pre_dict = pre_dict
        for eval_type, cfg in self.cfg.items():
            if cfg['effect']:
                result_dict.update(self.__getattribute__(eval_type)(cfg))
        return result_dict

    def parallel_eval_template(self, eval_func, eval_cfg):
        result = 0
        result_dict = {}
        metric_name = eval_func.__name__
        metric_name = re.sub(r'cal_', '', metric_name)
        write = eval_cfg.pop('write', False)
        for primary_key in self.gt_dict:
            gt_value, pre_value = self.gt_dict[primary_key][eval_cfg.get('gt_key')], self.pre_dict[primary_key][
                eval_cfg.get('pred_key')]
            result_item = eval_func(gt_value, pre_value, **eval_cfg.func_para)
            result += result_item
            if write:
                self.logger(f'{primary_key}: {metric_name}({result_item})')
            if self.csv_logger is not None:
                self.csv_logger[primary_key][metric_name] = result_item
        result_dict.update({f'{metric_name}': result / len(self.gt_dict)})
        return result_dict

    def dice_eval(self, cfg):
        return self.parallel_eval_template(cal_dice, cfg)


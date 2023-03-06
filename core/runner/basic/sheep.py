import os
from collections import abc as container_abc, defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import tqdm

from datasets.creat_dataloader import DataLoaderCreater
from evaluator.evaluator import Evaluator
from utils.distribution import sync_barrier_re1
from utils.loggers import LoggerManager
from utils.metrics import margin_of_error_at_confidence_level
from utils.utils import setup_device
from visualizer.vis_wrapper import Visualizer


class Sheep(object):
    """
    Include basic file system, test, print Evaluation.
    May be first brick of The Wall.
    """
    _runner_waiting = 0
    _runner_testing = 1

    def __init__(self, cfg, writer):
        self.runner_status = self._runner_waiting
        self.cfg, self.writer = cfg, writer
        self.logger = LoggerManager().get_logger()
        self.device = None
        self.init_device()
        self.occupyied_val = None
        if self.cfg.run.get('occupyied', False):
            self.occupyied_val = torch.rand((200, 24, 512, 512), device=self.device)
        self.dataLoaderCreater = None
        self.init_dataset()
        # set up test csv_logger
        self.construct_test_csv_logger()
        # Setup Evaluator
        self.pred_dict = {}
        self.evaluation_effect = self.cfg.get('evaluation', None) is not None \
                                 and self.cfg.evaluation.get('effect', False)
        if self.evaluation_effect:
            self.init_evaluator()
        self.status2str_mapping = {self._runner_testing: 'test', self._runner_waiting: 'wait'}
        self.test_set_weight = self.init_test_set_weight()

    def init_dataset(self):
        # Setup Dataloader
        self.dataLoaderCreater = DataLoaderCreater(self.cfg.dataset)
        for dataset_name, loader in self.dataLoaderCreater().items():
            self.__setattr__(f'{dataset_name}_loader', loader)
            self.logger(f'len(self.{dataset_name}_loader): {len(loader)}')

    def init_device(self):
        self.device = setup_device(self.cfg.gpus, mannul=True)

    def init_evaluator(self):
        # Prepare for test
        dataLoaderCreater = DataLoaderCreater(self.cfg['evaluation']['dataset'])
        for dataset_name, loader in dataLoaderCreater().items():
            self.__setattr__(f'{dataset_name}_eval_loader', loader)
        for eval_dataset_name in self.cfg['evaluation']['dataset']['dataset_name_list']:
            test_csv_logger = getattr(self, self.get_csv_logger_name(eval_dataset_name), None)
            self.__setattr__(f'{eval_dataset_name}_evaluator', Evaluator(
                gt_data_loader=self.__getattribute__(f'{eval_dataset_name}_eval_loader'),
                writer=self.writer, csv_logger=test_csv_logger,
                cfg=self.cfg['evaluation']['evaluator'],
                gt_need=self.cfg.evaluation.gt_need))

    @staticmethod
    def get_pred(data_dict):
        raise NotImplementedError

    def get_dataset_len(self, dataset_name):
        return len(self.__getattribute__(f'{dataset_name}_loader').dataset)

    def init_test_set_weight(self):
        if self.cfg.run.get('run_type', 'train') != 'test':
            valid_set_name_list = self.cfg.valid.get('valid_dataset', None)
            if valid_set_name_list is None or not valid_set_name_list:
                return None
            test_set_name_list = valid_set_name_list
        else:
            test_set_name_list = self.cfg.run.test_dataset
        name2len = dict()
        for set_name in test_set_name_list:
            set_len = self.get_dataset_len(set_name)
            name2len[set_name] = set_len
        total_set_len = sum(name2len.values())
        name2weight = {set_name: set_len / total_set_len for set_name, set_len in name2len.items()}
        return name2weight

    def construct_test_csv_logger(self):
        if not self.cfg.evaluation.get('save_csv_logger', False):
            return None, None
        test_dataset_name_list = self.cfg.run.test_dataset
        for test_set_name in test_dataset_name_list:
            test_csv_logger = defaultdict(lambda: {})
            csv_logger_name = self.get_csv_logger_name(test_set_name)
            csv_logger_path = self.get_csv_logger_save_path(test_set_name)
            self.__setattr__(csv_logger_name, test_csv_logger)
            save_csv_save_path = os.path.join(self.cfg.csv_dir, f'{csv_logger_name}.csv')
            self.__setattr__(csv_logger_path, save_csv_save_path)

    def write_eval(self, eval_info):
        out = '>> '
        mean_eval_dict = defaultdict(lambda: 0)
        for dataset_name, dataset_eval_info in eval_info.items():
            for item_name, item_value in dataset_eval_info.items():
                writed_name = f'{dataset_name}_{item_name}'
                out += f' {writed_name}: {item_value}'
                mean_eval_dict[item_name] += item_value * self.test_set_weight[dataset_name]
        self.logger(out)
        out = '>>'
        for eval_item, eval_value in mean_eval_dict.items():
            out += f'Mean {eval_item}: {eval_value}\n'
        self.logger(out)

    def after_test_batch_data(self, dataset_name, data_dict, for_pred_dict):
        if 'visualization' in self.cfg and self.cfg.visualization.get('effect', False):
            self.vis_batch_data(dataset_name, data_dict, for_pred_dict)

    def prepare_batch_show(self, data_dict):
        # primary_keys = data_dict[self.cfg.visualization.get('primary_key', 'img_name')]
        # primary_keys = petrificus_totalus(primary_keys)
        # show_key = self.cfg.visualization.get('showed_key', 'img')
        # batch_show_img = {name: t(img=petrificus_totalus(lorder_img)) for name, lorder_img, t in
        #                   zip(primary_keys, data_dict[show_key], data_dict['vis_img_transform'])}
        batch_show = data_dict['visualization']
        return batch_show

    def vis_batch_data(self, dataset_name, data_dict, for_pred_dict):
        batch_show = self.prepare_batch_show(data_dict)
        self.vis(dataset_name, batch_show, for_pred_dict)

    def prepare_test(self):
        pass

    def get_csv_logger_name(self, test_set_name):
        prefix = self.cfg.evaluation.get('csv_logger_prefix', '')
        return f'{prefix}_{test_set_name}_csv_logger'

    def get_csv_logger_save_path(self, test_set_name):
        prefix = self.cfg.evaluation.get('csv_logger_prefix', '')
        return f'{prefix}_{test_set_name}_csv_save_path'

    def finish_test_csv_logger(self):
        public_csv_logger = {}
        for test_set_name in self.cfg.run.test_dataset:
            test_csv_logger = getattr(self, self.get_csv_logger_name(test_set_name), None)
            if test_csv_logger is None:
                continue
            test_csv_save_path = getattr(self, self.get_csv_logger_save_path(test_set_name), None)
            self.logger(f'------------------------------------------------------------------\n'
                        f'Finish testing {test_set_name}:')
            self.finish_one_test_csv_logger(test_csv_logger)
            public_csv_logger.update({f'{test_set_name}_{key}': value for key, value in test_csv_logger.items()})
            dataF = self.get_dataFrame_from_test_csv_logger(test_csv_logger)
            self.save_csv_logger(dataF, test_csv_save_path)
        if public_csv_logger:
            self.logger(f'------------------------------------------------------------------\n'
                        f'Aggregate all csv:')
            self.finish_one_test_csv_logger(public_csv_logger)

    def finish_one_test_csv_logger(self, test_csv_logger):
        first_elem_key = list(test_csv_logger.keys())[0]
        first_elem_value = test_csv_logger[first_elem_key]
        metric_name_list = list(first_elem_value.keys())
        statistics_name_mapping = {'mean': np.mean, 'std': np.std,
                                   'margion_of_error_95': margin_of_error_at_confidence_level}
        make_latex = self.cfg.get('make_latex')
        is_make_latex = False
        if make_latex is not None and make_latex.get('effect', True):
            is_make_latex = True
            prefix_name = make_latex.get('prefix_name', 'mean')
            suffix_name = make_latex.get('suffix_name', 'std')
        for metric_name in metric_name_list:
            metric_value_list = []
            for csv_value in test_csv_logger.values():
                metric_v = csv_value.get(metric_name, None)
                if metric_v is not None:
                    metric_value_list.append(metric_v)
            for sta_name, sta_func in statistics_name_mapping.items():
                sta_value = sta_func(metric_value_list)
                self.logger(f'{sta_name} of {metric_name}: {sta_value}')
                if is_make_latex:
                    if sta_name == prefix_name:
                        prefix_value = sta_value
                    if sta_name == suffix_name:
                        suffix_value = sta_value
            if is_make_latex:
                self.logger(f'Metric {metric_name} latex format: {prefix_value:.3f}$\pm${suffix_value:.3f}')
            self.__setattr__(metric_name, prefix_value)

    def after_test(self, eval_info):
        self.write_eval(eval_info)
        self.finish_test_csv_logger()

    def test(self):
        eval_info = {}
        with torch.no_grad():
            for dataset_name in self.cfg.run.test_dataset:
                self.occupyied_val = None
                dataset_eval_info = self.test_specific_dataset(dataset_name)
                eval_info[dataset_name] = dataset_eval_info
        self.after_test(eval_info)

    def test_specific_dataset(self, dataset_name):
        self.runner_status = self._runner_testing
        pred_dict = {}
        self.prepare_test()
        dataloader = self.__getattribute__(f'{dataset_name}_loader')
        for batch_ind, data_dict in enumerate(
                tqdm.tqdm(dataloader, ncols=80, desc=f'testing {dataset_name}', position=0)):
            for_pred_dict = self.get_pred(data_dict)
            pred_dict.update(for_pred_dict)
            self.after_test_batch_data(dataset_name, data_dict, for_pred_dict)
        eval_info = self.__getattribute__(f'{dataset_name}_evaluator').evaluate(pred_dict)
        return eval_info

    def __call__(self):
        self.test()

    # def merge_eval_info(self, eval_info):
    #     dataset_len_list = []
    #     for dataset_name, dataset_eval_info in eval_info.items():
    #         dataset_len_list.append(len(self.__getattribute__(f'{dataset_name}_loader').dataset))
    #         for item_name, item_value in dataset_eval_info.items():
    #             x = 1
    #     return eval_info

    # ################# IO, need barrier ##############

    def get_dataFrame_from_test_csv_logger(self, test_csv_logger):
        first_elem_key = list(test_csv_logger.keys())[0]
        first_elem_value = test_csv_logger[first_elem_key]
        metric_name_list = list(first_elem_value.keys())
        dataf = {key: [] for key in ['primary_key'] + metric_name_list}
        for primary_key, metric_dict in test_csv_logger.items():
            dataf['primary_key'].append(primary_key)
            for metric_name, metric_value in metric_dict.items():
                dataf[metric_name].append(metric_value)
        return dataf

    @sync_barrier_re1
    def save_csv_logger(self, csv_logger, csv_save_path):
        pd_dataFrame = pd.DataFrame(csv_logger)
        pd_dataFrame.to_csv(csv_save_path, index=False)

    @sync_barrier_re1
    def vis(self, dataset_name, batch_img, pred_dict, batch_fixed_suffix=None):
        if isinstance(batch_img, dict):
            batch_img = batch_img[self.cfg.visualization.get('showed_key')]
        affi_name_pool = self.cfg.visualization.get('affi_name_pool', ['mask', 'hm', 'recist', 'bbox'])
        batch_affi_dict = deepcopy(pred_dict)
        for idx_in_batch, (img_name, img_affi_items) in enumerate(batch_affi_dict.items()):
            if isinstance(batch_img, container_abc.Mapping):
                cur_vis_img = batch_img[img_name]
            else:
                cur_vis_img = batch_img[idx_in_batch]
            if isinstance(cur_vis_img, container_abc.Mapping):
                cur_vis_img = cur_vis_img[self.cfg.visualization.get('showed_key', 'ori_img')]
            viser = Visualizer(f'{dataset_name}({img_name})', cur_vis_img, self.writer, self.cfg.visualization)
            affi_dict = {affi_name: affi for affi_name, affi in img_affi_items.items()
                         if any(map(lambda pool_name: pool_name in affi_name, affi_name_pool))}
            suffix = None if batch_fixed_suffix is None else batch_fixed_suffix
            if self.cfg.visualization.get('draw_gt', False):
                dataset_evaluator = self.__getattribute__(f'{dataset_name}_evaluator')
                gt_dict = dataset_evaluator.gt_dict[img_name]
                gt_affi_dict = {affi_name: affi for affi_name, affi in gt_dict.items()
                                if any(map(lambda pool_name: pool_name in affi_name, affi_name_pool))}
                if self.cfg.visualization.get('need_eval_info', False):
                    eval_info = dataset_evaluator.auto_evaluation(affi_dict, gt_affi_dict)
                    if suffix is None:
                        suffix = ''
                    for eval_key, eval_value in eval_info.items():
                        suffix = f'{suffix}_{eval_key}({eval_value:.4f})'
                viser.vis(gt_affi_dict, suffix=suffix)
            viser.vis(affi_dict, suffix=suffix)

    @sync_barrier_re1
    def tb_add_scalar(self, *args, **kwargs):
        self.writer.add_scalar(*args, **kwargs)

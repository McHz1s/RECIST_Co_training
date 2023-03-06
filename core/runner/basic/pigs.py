import asyncio
import gc
import os
import re
from collections import defaultdict

import numpy as np
import setproctitle
import torch
import torch.distributed as dist
import tqdm
from torch.cuda.amp import autocast, GradScaler

from core.loss import get_loss_function
from core.optimizer import DefaultOptimizerConstructor
from core.runner.basic.dogs import Dogs
from core.scheduler import get_scheduler
from utils.distribution import is_master_process, sync_barrier_re1
from utils.my_containers import AverageMeter, value_dict_mean
from utils.tensor_operator import petrificus_totalus, to_device


class Pigs(Dogs):
    """
    ~~Three Different Ones~~
    """
    _runner_training = 3
    _runner_validing = 4

    def __init__(self, *args, **kwargs):
        self.grad_scaler, self.optimizer, self.scheduler = [None] * 3
        self.start_epoch, self.epoch, self.n_step_global, self.lr = [1] * 2 + [0] * 2
        # trivial
        self.clean_his = False
        super(Pigs, self).__init__(*args, **kwargs)
        self.status2str_mapping.update({self._runner_training: 'train', self._runner_validing: 'valid'})
        if self.cfg.run.run_type == 'test':
            return
        # Setup loss function
        self.loss_name, self.loss_name_list = [None] * 2
        self.init_loss()
        # time-series
        self.time_series_name_list = ['epoch', 'n_step_global', 'lr']
        self.csv_logger, self.csv_save_path = self.construct_csv_logger()
        self.csv_cursor = 0
        # init valid
        self.valid_step_list = self.valid_init('train')
        # monitor
        self.monitor_name_list = self.cfg.run.monitor
        if self.monitor_name_list is None:
            self.monitor_name_list = []
        self.monitor = {f'best_{name}': 0 for name in self.monitor_name_list}
        self.save_epoch_list = self.cfg.run.get('save_epoch', [])
        if self.save_epoch_list is None:
            self.save_epoch_list = []
        if isinstance(self.save_epoch_list, str):
            self.save_epoch_list = eval(self.save_epoch_list)
        self.runner_status = self._runner_waiting
        self.clean_number = 3
        self.do_valid_this_epoch = False

    def init_network(self):
        super(Pigs, self).init_network()
        if self.cfg.run.run_type != 'test':
            self.init_optimizer()
            self.init_lr_scheduler()
            # Init time-series info

    def init_loss(self):
        # Setup loss function
        loss_cfg = self.cfg.run.loss.cfg
        self.loss_name = self.cfg.run.loss.name
        self.__setattr__(self.loss_name, get_loss_function(self.loss_name, loss_cfg))

    def init_optimizer(self):
        # Setup optimizer, lr_scheduler
        self.grad_scaler = GradScaler() if self.cfg.run.get('FP16', False) else None
        optim = DefaultOptimizerConstructor(self.cfg.run.get('optimizer', {}))
        self.optimizer = optim(self.network)

    def init_lr_scheduler(self):
        self.scheduler = get_scheduler(self.optimizer, self.cfg["run"]["lr_schedule"])

    def fetch_lr_from_optimizer(self):
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']

    def construct_csv_logger(self):
        csv_logger = defaultdict(list)
        csv_logger.update({monitor_name: [] for monitor_name in self.time_series_name_list})
        csv_save_path = os.path.join(self.cfg.csv_dir, 'time_series.csv')
        return csv_logger, csv_save_path

    def set_meter(self, meter_name_list):
        for name in meter_name_list:
            for prefix in [self.cfg.run.train_dataset] + self.cfg.valid.valid_dataset:
                self.__setattr__(f'{prefix}_{name}_meter', AverageMeter())

    def cal_and_update_loss(self, out_dict, target, dataset_name, backward=True):
        loss_dict = self.__getattribute__(self.loss_name)(out_dict, target)
        if self.loss_name_list is None:
            self.loss_name_list = loss_dict.keys()
            self.set_meter(self.loss_name_list)
        if backward:
            if self.grad_scaler is None:
                loss_dict['total_loss'].backward()
            else:
                self.grad_scaler.scale(loss_dict['total_loss']).backward()
                if self.cfg.run.get('grad_clip', -1) > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.run.get('grad_clip'))
        loss_dict = {loss_name: petrificus_totalus(loss_value) for loss_name, loss_value in loss_dict.items()}
        if self.is_distribution_running():
            gather_loss_dict = [None] * self.cfg.dist.world_size
            dist.all_gather_object(gather_loss_dict, loss_dict)
            loss_dict = value_dict_mean(gather_loss_dict)
        for loss_name, loss_value in loss_dict.items():
            self.__getattribute__(f'{dataset_name}_{loss_name}_meter').update(petrificus_totalus(loss_value))

    def fp16_optimizer_update(self):
        fp16 = self.cfg.get('fp16', False)
        if fp16:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

    def write_loss(self, dataset_name):
        out = '>> '
        for loss_name in self.loss_name_list:
            prefix_name = f'{dataset_name}_{loss_name}'
            meter = self.__getattribute__(f'{prefix_name}_meter')
            loss_scalar = np.round(meter.avg, 4)
            self.tb_add_scalar(prefix_name, loss_scalar, self.n_step_global)
            out += f' {prefix_name}: {loss_scalar}'
            self.csv_logger[prefix_name].append(loss_scalar)
            meter.reset()
        self.logger(out)

    def write_eval(self, eval_info):
        if self.cfg.run.run_type == 'test':
            super(Pigs, self).write_eval(eval_info)
            return
        out = '>> '
        for dataset_name, dataset_eval_info in eval_info.items():
            for item_name, item_value in dataset_eval_info.items():
                writed_name = f'{dataset_name}_{item_name}'
                out += f' {writed_name}: {item_value}'
                self.csv_logger[writed_name].append(item_value)
                self.tb_add_scalar(writed_name, item_value, self.n_step_global)
        self.logger(out)

    def update_csv_logger_time_series(self):
        for name in self.time_series_name_list:
            self.csv_logger[name].append(self.__getattribute__(name))

    def update_monitor(self, eval_info):
        is_best = False
        eval_item_dict = defaultdict(lambda: 0)
        for dataset_name, dataset_eval_info in eval_info.items():
            for eval_name, eval_value in dataset_eval_info.items():
                eval_item_dict[eval_name] += eval_value
        for eval_name, eval_value in eval_item_dict.items():
            eval_value /= len(eval_info)
            self.csv_logger[f'mean_{eval_name}'].append(eval_value)
            if f'best_{eval_name}' in self.monitor:
                if eval_value > self.monitor[f'best_{eval_name}']:
                    is_best = True
                    self.monitor[f'best_{eval_name}'] = eval_value
                self.csv_logger[f'best_{eval_name}'].append(self.monitor[f'best_{eval_name}'])
        return is_best

    @sync_barrier_re1
    def torch_save(self, state, path):
        saving_work = self.asycn_torch_save(state, path)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(saving_work)

    @staticmethod
    async def asycn_torch_save(state, path):
        torch.save(state, path)

    def save_by_name(self, name):
        state_name_list = ['network', 'optimizer', 'scheduler']
        state = {save_name: self.__getattribute__(save_name).state_dict() for save_name in state_name_list}
        scalar_name_list = ['epoch']
        state.update({scalar_name: self.__getattribute__(scalar_name) for scalar_name in scalar_name_list})
        self.torch_save(state, os.path.join(self.cfg.weight_dir, name))

    def save_best(self):
        monitor_pair = ''
        for key, values in self.monitor.items():
            monitor_pair += f'{key}({values:4f})_'
        save_name = f'{self.cfg.network.name}_{monitor_pair}_epoch({self.epoch}).pth.tar'
        self.save_by_name(save_name)

    def resume(self):
        if 'convert_pretrained' in self.cfg.network:
            self.cfg.network.convert_pretrained = None
            self.network.cfg.convert_pretrained = None
        if self.cfg.run.run_type == 'test':
            return super(Pigs, self).resume()
        if os.path.isfile(self.cfg["run"]["resume"]):
            checkpoint = torch.load(self.cfg["run"]["resume"], map_location='cpu')
            self.network.load_pretrained_params(None, True, checkpoint)
            self.start_epoch = checkpoint["epoch"] + 1
            checkpoint.pop('network')
            checkpoint.pop('epoch')
            for state_name, state in checkpoint.items():
                self.__getattribute__(state_name).load_state_dict(state)
            self.logger(
                "Loading checkpoint '{}' (epoch {})".format(self.cfg["run"]["resume"], self.start_epoch - 1))
        else:
            self.logger("No checkpoint found at '{}'".format(self.cfg["run"]["resume"]))
        self.epoch = self.start_epoch
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
        self.clean_his = True
        self.clean_number = 10

    def make_in_out_target(self, data_dict, need_target=True):
        re_dict = defaultdict(dict)
        device_mapping = {'gpu': self.device, 'cpu': 'cpu', 'identity': 'identity'}
        for need_dict_name in self.input_target_dict_name_list:
            device, in_or_out = need_dict_name.split('_')[:2]
            device = device_mapping[device]
            if not need_target and in_or_out == 'target':
                continue
            for data_type, data_name_list in self.__getattribute__(need_dict_name).items():
                for data_name in data_name_list:
                    re_dict[in_or_out][data_name] = to_device(data_dict[data_name], device, data_type)
        return re_dict

    def clean_cache(self):
        if self.clean_his or self.clean_number > 0:
            self.clean_number -= 1
            gc.collect()
            torch.cuda.empty_cache()

    def train_step(self, data_dict):
        self.run_model.train()
        # todo can not be write as this
        self.network.post_process_effect = False
        # Prepare input
        inp_target = self.make_in_out_target(data_dict)
        inp, target = inp_target['input'], inp_target['target']
        self.optimizer.zero_grad()
        self.clean_cache()
        if self.grad_scaler is None:
            out_dict = self.run_model(inp)
            self.cal_and_update_loss(out_dict, target, self.cfg.run.train_dataset)
            self.optimizer.step()
        else:
            with autocast():
                out_dict = self.run_model(inp)
                self.cal_and_update_loss(out_dict, target, self.cfg.run.train_dataset)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

    def valid_specific_dataset(self, dataset_name, metric=True):
        pred_dict = {}
        eval_info = None
        dataloader = self.__getattribute__(f'{dataset_name}_loader')
        for batch_ind, data_dict in enumerate(dataloader):
            # Prepare input
            inp_target = self.make_in_out_target(data_dict)
            inp, target = inp_target['input'], inp_target['target']
            out_dict = self.FP16_network_forward(self.cfg.valid.get('FP16', False), inp)
            if metric:
                for_pred_dict = {name: item for name, item in out_dict.items() if 'out' not in name}
                if self.is_distribution_running():
                    for_pred_dict = self.ddp_gather_pred_dict(for_pred_dict)
                pred_dict.update(for_pred_dict)
            self.cal_and_update_loss(out_dict, target, dataset_name, backward=False)
        if metric:
            eval_info = self.evaluate(pred_dict, dataset_name)
        self.write_loss(dataset_name)
        return eval_info

    def valid(self):
        metric = self.cfg['evaluation']['effect'] and self.epoch >= self.cfg.valid.get('do_metric_delay', -1)
        self.runner_status = self._runner_validing
        self.network.switch_network_status('validing')
        self.run_model.eval()
        eval_info = {}
        if metric:
            self.network.post_process_effect = True
        with torch.no_grad():
            for dataset_name in self.cfg.valid.valid_dataset:
                dataset_eval_info = self.valid_specific_dataset(dataset_name, metric)
                if eval_info is not None:
                    eval_info[dataset_name] = dataset_eval_info
        self.network.post_process_effect = False
        return eval_info

    def valid_init(self, data_type):
        n_valid_per_epoch = self.cfg['valid']['n_valid_per_epoch']
        if n_valid_per_epoch > 0:
            train_len = self.train_loader_len
            valid_step_list = list(
                np.linspace(0, train_len, num=n_valid_per_epoch + 1, endpoint=True))[1:]
            valid_step_list = [int(step) for step in valid_step_list]
            self.logger('valid_step_list {}'.format(valid_step_list))
        else:
            valid_step_list = None
            self.logger('No validation')
        return valid_step_list

    def before_train_epoch(self, data_type):
        self.do_valid_this_epoch = False
        if self.is_distribution_running():
            for dataset_name in self.cfg.run.train_dataset:
                self.__getattribute__(f'{dataset_name}_loader').sampler.set_epoch(self.epoch)

    def before_train_step(self, dataset_name):
        self.runner_status = self._runner_training
        self.network.switch_network_status('training')
        if self.occupyied_val is not None:
            self.occupyied_val = None
            torch.cuda.empty_cache()

    def run(self):
        # ###################################################################################################################
        #  epoch
        # ###################################################################################################################
        while self.epoch <= self.cfg['run']['n_epoch']:
            # ###############################################################################################################
            #  train
            # ###############################################################################################################
            train_dataset = self.cfg.run.train_dataset
            self.before_train_epoch(train_dataset)
            self.logger(f'>> Task Name : {self.cfg.run.task}')
            self.logger(f'>> Epoch[{int(self.epoch)}/{int(self.cfg["run"]["n_epoch"])}]')
            train_dataset = self.cfg.run.train_dataset
            data_loader_list = [self.__getattribute__(f'{set_name}_loader') for set_name in train_dataset]
            batch_num = 0
            for data_loader in data_loader_list:
                for batch_ind, data_dict in enumerate(tqdm.tqdm(data_loader, desc='training', ncols=80)):
                    self.before_train_step(train_dataset)
                    self.train_step(data_dict)
                    self.after_train_step(batch_num)
                    batch_num += 1
            self.after_train_epoch()
        self.after_train()

    def after_train(self):
        if is_master_process():
            self.writer.close()

    def whether_to_validate(self, n_step):
        cond_1 = self.valid_step_list is not None and n_step in self.valid_step_list
        valid_epoch_interval = self.cfg.valid.get('valid_epoch_interval', 1)
        cond_3 = self.epoch % valid_epoch_interval == 0
        valid_epoch_delay = self.cfg.valid.get('valid_epoch_delay', 0)
        cond_4 = self.epoch >= valid_epoch_delay
        return cond_1 and cond_3 and cond_4

    @property
    def train_loader_len(self):
        return sum(map(lambda x: len(self.__getattribute__(f'{x}_loader')), self.cfg.run.train_dataset))

    def after_train_step(self, batch_ind):
        n_step = batch_ind + 1
        self.fetch_lr_from_optimizer()
        self.n_step_global = int((self.epoch - 1) * self.train_loader_len + n_step)
        # ###########################################################################################################
        #  validation
        # ###########################################################################################################
        if self.whether_to_validate(n_step):
            print()
            self.do_valid_this_epoch = True
            self.update_csv_logger_time_series()
            self.write_loss(self.cfg.run.train_dataset)
            eval_info = self.valid()
            if eval_info is not None:
                self.write_eval(eval_info)
                if self.update_monitor(eval_info) and self.cfg.run.get('save_best', True):
                    self.save_best()
            self.update_proctitle()
            self.save_csv_logger(self.csv_logger, self.csv_save_path)

    def update_proctitle(self):
        proctitle_text = f'{self.cfg.run.task.split("/")[-1]}:E({self.epoch})'
        if self.monitor:
            for best_monitor_name, monitor_value in self.monitor.items():
                monitor_name = re.sub(r'best_', '', best_monitor_name)
                proctitle_text += f'_{monitor_name}({self.csv_logger[best_monitor_name][-1]:.4f})'
        setproctitle.setproctitle(proctitle_text)

    def save_epoch(self):
        first_str = f'epoch({self.epoch:04d})'
        last_str_list = ['']
        for monitor_name in self.monitor_name_list:
            if monitor_name in self.csv_logger:
                last_str_list.append(f'{monitor_name}({self.csv_logger[monitor_name][-1]})')
        name = '_'.join([first_str] + last_str_list) + 'pth.tar'
        self.save_by_name(name)

    def after_train_epoch(self):
        if not self.do_valid_this_epoch:
            self.update_csv_logger_time_series()
            self.write_loss(self.cfg.run.train_dataset)
            self.save_csv_logger(self.csv_logger, self.csv_save_path)
        if self.epoch in self.save_epoch_list:
            self.save_epoch()
        self.save_by_name("last_checkpoint.pth.tar")
        self.epoch += 1
        self.scheduler.step()
        if self.cfg.run.get('clean_gpu_each_epoch', False):
            self.clean_his = True
        self.clean_cache()
        self.clean_his = False

    def __call__(self):
        self.__getattribute__(self.cfg.run.get('run_type', 'run'))()

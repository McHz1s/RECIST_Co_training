import importlib
from types import FunctionType
from typing import Dict

import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as torchDatasetBasic
from torch.utils.data.distributed import DistributedSampler

from datasets.utils.collate import default_collate as dc
from utils.my_containers import Register, deep_dict_get

instanceRegister, taskAbsRegister = Register(), Register()


class DataLoaderCreater(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_pool = {}

    def create_general_inherit_dataset_instance(self, dataset_cfg):
        Instance = instanceRegister[dataset_cfg.instance.name]
        TaskAbs = taskAbsRegister[dataset_cfg.taskAbs.name]

        if type(TaskAbs) == FunctionType:
            TaskAbs = TaskAbs(dataset_cfg.taskAbs)

        class GeneralDatasetInstance(Instance, TaskAbs):
            def __init__(self, cfg):
                super(GeneralDatasetInstance, self).__init__(cfg.instance)
                TaskAbs.__init__(self, cfg.taskAbs)

        return GeneralDatasetInstance(dataset_cfg)

    def create_Composition_Instance(self, dataset_cfg):
        component_dataset_name_list = dataset_cfg.component_dataset_name_list
        component_dataset_dict = {d_name: self.create_dataset(d_name, dataset_cfg[d_name])
                                  for d_name in component_dataset_name_list}

        class CompositionDatasetInstance(torchDatasetBasic):
            index2offset: Dict[int, int]
            composition_len: int
            index2datasetName: Dict[int, str]
            index2dataset: Dict[int, torchDatasetBasic]

            def __init__(self, cfg):
                super(CompositionDatasetInstance, self).__init__()
                self.cfg = cfg
                self.component_dataset_name_list = cfg.component_dataset_name_list
                self.init_component_dataset()
                self.component_index_list = list(self.index2datasetName.keys())

            def init_component_dataset(self):
                composition_len = 0
                index2dataset, index2dataset_name, index2offset = [{} for _ in range(3)]
                for dataset_name in self.component_dataset_name_list:
                    component_dataset = component_dataset_dict[dataset_name]
                    self.__setattr__(dataset_name, component_dataset)
                    index2dataset.update({i: component_dataset
                                          for i in range(composition_len, composition_len + len(component_dataset))})
                    index2dataset_name.update({i: dataset_name
                                               for i in
                                               range(composition_len, composition_len + len(component_dataset))})
                    index2offset.update({i: composition_len
                                         for i in range(composition_len, composition_len + len(component_dataset))})
                    composition_len += len(component_dataset)
                self.composition_len = composition_len
                self.index2dataset = index2dataset
                self.index2datasetName = index2dataset_name
                self.index2offset = index2offset

            def __len__(self):
                return self.composition_len

            def refine_datadict(self, dataset_name, datadict):
                for key, value in datadict.items():
                    if key in self.cfg.get('need_refine_key_list', ['id', 'img_name']):
                        datadict[key] = f'{dataset_name}_{value}'

            def __getitem__(self, index):
                dataset = self.index2dataset[index]
                offset = self.index2offset[index]
                dataset_idx = index - offset
                datadict = dataset[dataset_idx]
                self.refine_datadict(self.index2datasetName[index], datadict)
                return datadict

        return CompositionDatasetInstance(dataset_cfg)

    def build_from_path(self, dataset_cfg):
        module_path = dataset_cfg.module_path
        name = dataset_cfg.name
        path = f'datasets.instance.{module_path}'
        dataset_module = importlib.import_module(path)
        dataset = getattr(dataset_module, name)
        return dataset(dataset_cfg)

    def create_dataset(self, dataset_name, dataset_cfg):
        if dataset_name not in self.dataset_pool:
            build_method = dataset_cfg.get('build_method', 'create_general_inherit_dataset_instance')
            dataset = self.__getattribute__(build_method)(dataset_cfg)
            self.dataset_pool[dataset_name] = dataset
        else:
            dataset = self.dataset_pool[dataset_name]
        return dataset

    def __call__(self, cfg=None):
        loader_dict = {}
        if cfg is None:
            cfg = self.cfg
        dataset_name_list = cfg['dataset_name_list']
        for dataset_name in dataset_name_list:
            dataset_cfg = cfg[dataset_name]
            dataset = self.create_dataset(dataset_name, dataset_cfg)
            dataloader_need = ['batch_size', 'n_workers']
            batch_size, num_workers = tuple(
                map(lambda x: dataset_cfg.dataloader.get(x[0], x[1]), zip(dataloader_need, [1, 0])))
            run_mode = deep_dict_get(dataset_cfg, 'run_mode', 'train')
            shuffle = True if run_mode == 'train' else False
            cf = dc if run_mode != 'analyse' else lambda x: x
            sampler = None
            if 'dist' in cfg and run_mode in ['train', 'valid', 'test']:
                shuffle = False
                sampler = DistributedSampler(dataset,
                                             num_replicas=cfg.dist.world_size, rank=cfg.dist.rank,
                                             shuffle=True if run_mode == 'train' else False)
            pin_memory = True if torch.cuda.current_device() != 0 else False
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                drop_last=True if run_mode == 'train' else False, collate_fn=cf, pin_memory=pin_memory,
                                sampler=sampler)
            if dataset_cfg.get('prepare_datase_cache', False) and dataset_cfg.prepare_datase_cache.effect:
                batch_size, num_workers = tuple(
                    map(lambda x: dataset_cfg.prepare_datase_cache.get(x[0], x[1]), zip(dataloader_need, [1, 0])))
                prepare_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                            collate_fn=cf, sampler=sampler)
                for _ in tqdm.tqdm(prepare_loader, desc=f'Preparing {dataset_name} dataset cache...', ncols=80,
                                   position=0):
                    pass
            loader_dict[dataset_name] = loader
        return loader_dict

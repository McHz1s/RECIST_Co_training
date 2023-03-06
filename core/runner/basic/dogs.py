import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from core.runner.basic.dog import Dog
from utils.my_containers import merge_dis_dict


class Dogs(Dog):
    def __init__(self, *args, **kwargs):
        super(Dogs, self).__init__(*args, **kwargs)
        # fix logger bug caused by import nni
        if self.is_distribution_running():
            self.local_rank = self.cfg.dist.rank
            self.run_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.run_model)
            self.run_model = DistributedDataParallel(self.run_model,
                                                     device_ids=[torch.cuda.current_device()],
                                                     find_unused_parameters=True)

    def is_distribution_running(self):
        return 'dist' in self.cfg

    def ddp_gather_pred_dict(self, for_pred_dict):
        gather_for_pred_dict = [None] * self.cfg.dist.world_size
        dist.all_gather_object(gather_for_pred_dict, for_pred_dict)
        for_pred_dict = merge_dis_dict(gather_for_pred_dict)
        return for_pred_dict

    def get_pred(self, data_dict):
        for_pred_dict = super(Dogs, self).get_pred(data_dict)
        if self.is_distribution_running():
            for_pred_dict = self.ddp_gather_pred_dict(for_pred_dict)
        return for_pred_dict

    def prepare_batch_show(self, data_dict):
        batch_show = super(Dogs, self).prepare_batch_show(data_dict)
        if self.is_distribution_running():
            batch_show = self.ddp_gather_pred_dict(batch_show)
            dist.barrier()
        return batch_show

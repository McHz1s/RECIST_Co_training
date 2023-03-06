from torch.utils.tensorboard import SummaryWriter

from utils.distribution import is_master_process, sync_barrier_re1
from utils.my_containers import singleton


@singleton
class SummaryWriterManager(object):
    def __init__(self, *args, **kwargs):
        if not is_master_process():
            return
        self.writer = SummaryWriter(*args, **kwargs)
        for func_name in dir(self.writer):
            if func_name.startswith('_'):
                continue
            self.__setattr__(func_name, sync_barrier_re1(self.writer.__getattribute__(func_name)))

    # def __getattribute__(self, item):
    #     if 'add' in item or item in ['close', 'flush']:
    #         return sync_barrier_re1(self.writer.__getattribute__(item))



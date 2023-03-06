import os
import torch
from torch import distributed as dist
from functools import partial


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def is_master_process():
    return os.environ.get('RANK', '-1') in ['-1', '0']


def is_single_process():
    return os.environ.get('RANK', '-1') == '-1'


def sync_barrier_reX(func, X):
    """
    Warning: This is a very CASUAL implementation,
    Nesting doll of this function (e.g., recursion_ will lead to DEAD LOCK!
    Wrapper this func X.
    :param func: function only effective in master process, must be atomic operation.
    :param X: format of func returning to create fake return for sub-process
    :return: a barrier func
    """
    def barrier_func(*args, **kwarg):
        if is_single_process():
            return func(*args, **kwarg)
        x = X
        if is_master_process():
            x = func(*args, **kwarg)
            dist.barrier()
        else:
            dist.barrier()
        return x

    return barrier_func


sync_barrier_re1 = partial(sync_barrier_reX, X=None)
sync_barriers_re2 = partial(sync_barrier_reX, X=[None, None])

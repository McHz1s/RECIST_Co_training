import torch.nn as nn


class BasicPostProcess(nn.Module):
    def __init__(self, post_process_cfg):
        super(BasicPostProcess, self).__init__()
        self.cfg = post_process_cfg



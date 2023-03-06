from utils.my_containers import Register

model_register = Register()
from .mstfamily.MSTfamily import MSTfamily
from .unet.UNet import UNet
from .arunet.ARUNet import ARUNet
from .hnn.HNN import HNN


def get_model(model_name, model_cfg):
    return model_register[model_name](model_cfg)

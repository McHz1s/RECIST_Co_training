from utils.tensor_operator import petrificus_totalus


def merge_loss_dict(loss_dict_list, loss_name_list=None, weight_dict=None):
    re_loss_dict = dict()
    re_total_loss = 0
    for loss_i, loss_dict in enumerate(loss_dict_list):
        total_loss = loss_dict.pop('total_loss')
        if loss_name_list is not None and weight_dict is not None:
            weight = weight_dict[loss_name_list[loss_i]]
        else:
            weight = 1
        re_total_loss = re_total_loss + weight * total_loss
        re_loss_dict.update(loss_dict)
    re_loss_dict['total_loss'] = re_total_loss
    return re_loss_dict


def loss_dict_remake_wrapper(loss_func):
    def remake_loss_dict(*args, **kwargs):
        loss_dict = loss_func(*args, **kwargs)
        total_loss = loss_dict.pop('total_loss')
        new_loss_dict = {'total_loss': total_loss}
        for loss_name, loss_value in loss_dict.items():
            new_loss_dict[loss_name] = petrificus_totalus(loss_value)
        del loss_dict
        return new_loss_dict

    return remake_loss_dict

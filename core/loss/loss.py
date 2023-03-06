import re

import torch
import torch.nn as nn

from utils.my_containers import Register

ATOMLOSS = Register()


class FullSupervisedLoss(nn.Module):
    def __init__(self, cfg):
        super(FullSupervisedLoss, self).__init__()
        self.cfg = cfg
        self.supv_loss_list = cfg.supv_loss
        if type(self.supv_loss_list) == str:
            self.supv_loss_list = [self.supv_loss_list]
        self.supv_loss_func_list = [ATOMLOSS[supv_loss]() for supv_loss in self.supv_loss_list]

    def forward(self, out: dict, target: dict):
        supv_loss_dict = {}
        supv_loss = 0
        mask = target['supv_mask']
        prob_map = out['prob_map']
        for supv_loss_name, supv_loss_func, supv_loss_weight in \
                zip(self.supv_loss_list, self.supv_loss_func_list, self.cfg.supv_loss_weight_list):
            supv_loss_part = supv_loss_func(prob_map, mask)
            supv_loss_dict[supv_loss_name] = supv_loss_part
            supv_loss += supv_loss_weight * supv_loss_part
        total = supv_loss
        loss_dict = {'total_loss': total}
        loss_dict.update(supv_loss_dict)
        return loss_dict


class PseudoSupLoss(nn.Module):
    def __init__(self, cfg):
        super(PseudoSupLoss, self).__init__()
        self.cfg = cfg
        self.supv_loss_func = ATOMLOSS[cfg.supv_loss]()

    def forward(self, model_out_dict: dict, target: dict):
        """
        supv = supervised
        Args:
            model_out_dict: {'model_0_out':..., 'model_1_out': ...}
            target: {diamond_mask:..., circle_mask:...}

        Returns:
            dict:{each_model_name_sup_loss:...
                    cons_{model_id}-{model_id}_loss}

        """
        model_out_dict = {key: values for key, values in model_out_dict.items() if 'out' in key}
        supv_name_list = [re.sub(r'out', 'supv_loss', key) for key in model_out_dict.keys()]
        supv_loss_dict = dict(zip(supv_name_list, [0] * self.cfg.model_num))
        supv_loss = 0
        for i, (out_name, out) in enumerate(model_out_dict.items()):
            psueodo_mask = list(target['psuedo_mask_dict'].values())[i]
            prob_map = out['prob_map']
            supv_loss_part = self.supv_loss_func(prob_map, psueodo_mask)
            supv_loss_dict[supv_name_list[i]] += supv_loss_part
            supv_loss += supv_loss_part
        supv_loss /= self.cfg.model_num
        total = supv_loss
        loss_dict = {'total_loss': total, 'supv_loss': supv_loss}
        loss_dict.update(supv_loss_dict)
        return loss_dict


class CoSegLoss(PseudoSupLoss):
    """
    Consist Loss in uncertain region
    """

    def __init__(self, cfg):
        super(CoSegLoss, self).__init__(cfg)
        self.consistent_loss_func = ATOMLOSS[cfg.cons_loss]()
        self.cons_loss_name_list = []
        for i in range(self.cfg.model_num):
            for j in range(i + 1, self.cfg.model_num):
                self.cons_loss_name_list.append(f'model_{i}vs{j}_cons_loss')

    def forward(self, model_out_dict: dict, target: dict):
        """
        supv = supervised
        Args:
            model_out_dict: {'model_0_out':..., 'model_1_out': ...}
            target: {diamond_mask:..., circle_mask:...}

        Returns:
            dict:{each_model_name_sup_loss:...
                    cons_{model_id}-{model_id}_loss}

        """
        model_out_dict = {key: values for key, values in model_out_dict.items() if 'out' in key}
        supv_name_list = [re.sub(r'out', 'supv_loss', key) for key in model_out_dict.keys()]
        model_name_list = [re.sub(r'_out', '', key) for key in model_out_dict.keys()]
        supv_loss_dict = dict(zip(supv_name_list, [0] * self.cfg.model_num))
        cons_loss_dict = dict(zip(self.cons_loss_name_list, [0] * len(self.cons_loss_name_list)))
        supv_loss, cons_loss = 0, 0
        cons_cur = 0
        for i, (out_name, out) in enumerate(model_out_dict.items()):
            psueodo_mask = list(target['psuedo_mask_dict'].values())[i]
            shield_mask = target['shield_mask']
            prob_map = out['prob_map']
            zero_tensor = torch.zeros_like(prob_map)
            supv_loss_part = self.supv_loss_func(prob_map, psueodo_mask)
            supv_loss_dict[supv_name_list[i]] += supv_loss_part
            supv_loss += supv_loss_part
            for j in range(i + 1, len(model_out_dict)):
                com_prob_map = model_out_dict[f'{model_name_list[j]}_out']['prob_map']
                com0 = torch.where(shield_mask == 2, prob_map, zero_tensor)
                com1 = torch.where(shield_mask == 2, com_prob_map, zero_tensor)
                cons_loss_part = self.consistent_loss_func(com0, com1)
                cons_loss_dict[self.cons_loss_name_list[cons_cur]] += cons_loss_part
                cons_loss += cons_loss_part
                cons_cur += 1
        supv_loss /= self.cfg.model_num
        if (self.cfg.model_num - 1) * self.cfg.model_num // 2 != 0:
            cons_loss /= (self.cfg.model_num - 1) * self.cfg.model_num // 2
        cons_weight = 0 if supv_loss > self.cfg.cons_delay else self.cfg.weight[1]
        total = self.cfg.weight[0] * supv_loss + cons_weight * cons_loss
        loss_dict = {'total_loss': total, 'supv_loss': supv_loss, 'cons_loss': cons_loss}
        loss_dict.update(supv_loss_dict)
        loss_dict.update(cons_loss_dict)
        return loss_dict


class HeadCoSegLoss(CoSegLoss):
    def __init__(self, cfg):
        super(HeadCoSegLoss, self).__init__(cfg)
        for i, _ in enumerate(self.cons_loss_name_list):
            self.cons_loss_name_list[i] = re.sub(r'model', '', self.cons_loss_name_list[i])

    def forward(self, head_out_dict: dict, target: dict):
        """
        supv = supervised
        Args:
            model_out_dict: {'model_0_out':..., 'model_1_out': ...}
            target: {diamond_mask:..., circle_mask:...}

        Returns:
            dict:{each_model_name_sup_loss:...
                    cons_{model_id}-{model_id}_loss}

        """
        supv_name_list = [re.sub(r'prob_map', 'supv_loss', key) for key in head_out_dict.keys()]
        head_num = len(supv_name_list)
        head_name_list = [re.sub(r'_prob_map', '', key) for key in head_out_dict.keys()]
        supv_loss_dict = dict(zip([f'_{i}_supv_loss' for i in range(head_num)], [0] * head_num))
        cons_loss_dict = dict(zip(self.cons_loss_name_list, [0] * len(self.cons_loss_name_list)))
        supv_loss, cons_loss = 0, 0
        cons_cur = 0
        for i, (out_name, prob_map) in enumerate(head_out_dict.items()):
            psueodo_mask = list(target['psuedo_mask_dict'].values())[i]
            shield_mask = target['shield_mask']
            zero_tensor = torch.zeros_like(prob_map)
            supv_loss_part = self.supv_loss_func(prob_map, psueodo_mask)
            supv_loss_dict[f'_{i}_supv_loss'] += supv_loss_part
            supv_loss += supv_loss_part
            for j in range(i + 1, len(head_out_dict)):
                com_prob_map = head_out_dict[f'{head_name_list[j]}_prob_map']
                com0 = torch.where(shield_mask == 2, prob_map, zero_tensor)
                com1 = torch.where(shield_mask == 2, com_prob_map, zero_tensor)
                cons_loss_part = self.consistent_loss_func(com0, com1)
                cons_loss_dict[self.cons_loss_name_list[cons_cur]] += cons_loss_part
                cons_loss += cons_loss_part
                cons_cur += 1
        supv_loss /= head_num
        if (head_num - 1) * head_num // 2 != 0:
            cons_loss /= (head_num - 1) * head_num // 2
        cons_weight = 0 if supv_loss > self.cfg.cons_delay else self.cfg.weight[1]
        total = self.cfg.weight[0] * supv_loss + cons_weight * cons_loss
        loss_dict = {'total_loss': total, 'supv_loss': supv_loss, 'cons_loss': cons_loss}
        loss_dict.update(supv_loss_dict)
        loss_dict.update(cons_loss_dict)
        return loss_dict


class MultiSegHeadLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiSegHeadLoss, self).__init__()
        self.cfg = cfg
        self.single_head_loss = globals()[cfg.single_head_loss.name](cfg.single_head_loss.cfg)

    def forward(self, output_dict, target_dict):
        model_name_list = list(output_dict)
        model_name_list = [re.sub(r'_out', '', name) for name in model_name_list if '_out' in name]
        out_name_list = list(output_dict[f'{model_name_list[0]}_out'])
        head_name_list = [re.sub(r'_prob_map', '', name) for name in out_name_list if '_prob_map' in name]
        total = 0
        mul_out_dict = {}
        for h_i, head_name in enumerate(head_name_list):
            single_head_out_dict = {
                f'{model_name}_out': {'prob_map': output_dict[f'{model_name}_out'][f'{head_name}_prob_map']}
                for model_name in model_name_list}
            head_loss_dict = self.single_head_loss(single_head_out_dict, target_dict)
            model_head_loss_dict = {f'{head_name}_{key}': value for key, value in head_loss_dict.items()}
            mul_out_dict.update(model_head_loss_dict)
            total += self.cfg.weight[h_i] * head_loss_dict['total_loss']
        mul_out_dict['total_loss'] = total
        return mul_out_dict

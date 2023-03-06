import torch.nn as nn

from .encoder_decoder import EncoderDecoder
from .. import MSTfamily
from ..MSTfamily import SEGMENTORS


@SEGMENTORS.register_module()
class MutiHeadEncoderDecoder(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self, head_num=2, *args, **kwargs):
        self.head_num = head_num
        super(MutiHeadEncoderDecoder, self).__init__(*args, **kwargs)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        super(MutiHeadEncoderDecoder, self)._init_decode_head(decode_head)
        self.extra_decode_head_name_list, self.extra_decode_head_list = [[] for _ in range(2)]
        for i in range(self.head_num - 1):
            name = f'{decode_head.type}_{i}'
            self.extra_decode_head_name_list.append(name)
            self.extra_decode_head_list.append(MSTfamily.build_head(decode_head))
        self.extra_decode_head_list = nn.ModuleList(self.extra_decode_head_list)

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                # TODO: not implement this part to muti head
                raise NotImplementedError
                # self.aux_head_name = 'Sequence_head'
                # self.auxiliary_head = nn.ModuleList()
                # for head_cfg in auxiliary_head:
                #     self.auxiliary_head.append(MSTfamily.build_head(head_cfg))
                #     self.aux_head_name = f'{auxiliary_head}_{head_cfg.type}'
            else:
                self.aux_head_name = auxiliary_head.type
                self.auxiliary_head = MSTfamily.build_head(auxiliary_head)
                extra_auxiliary_head_list = []
                for i in range(self.head_num - 1):
                    extra_auxiliary_head_list.append(MSTfamily.build_head(auxiliary_head))
                self.extra_auxiliary_head_list = nn.ModuleList(extra_auxiliary_head_list)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(MutiHeadEncoderDecoder, self).init_weights(pretrained)
        for each_extra_decode_head, each_extra_auxiliary_head in \
                zip(self.extra_decode_head_list, self.extra_auxiliary_head_list):
            each_extra_decode_head.load_state_dict(self.decode_head.state_dict())
            each_extra_auxiliary_head.load_state_dict(self.auxiliary_head.state_dict())

    def forward(self, img):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `./datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        feat_out_dict = self.extract_feat(img)

        out_dict = feat_out_dict

        decode_out_dict = self.decode_head(feat_out_dict['backbone_feat'])
        out_dict[f'{self.decode_head_name}_0_prob_map'] = decode_out_dict['prob_map']
        for i, decode_head in enumerate(self.extra_decode_head_list):
            decode_out_dict = decode_head(feat_out_dict['backbone_feat'])
            out_dict[f'{self.decode_head_name}_{i + 1}_prob_map'] = decode_out_dict['prob_map']
        if self.with_auxiliary_head:
            aux_out_dict = self.auxiliary_head(feat_out_dict['backbone_feat'])
            out_dict[f'{self.aux_head_name}_0_prob_map'] = aux_out_dict['prob_map']
            for i, auxiliary_head in enumerate(self.extra_auxiliary_head_list):
                aux_out_dict = auxiliary_head(feat_out_dict['backbone_feat'])
                out_dict[f'{self.aux_head_name}_{i + 1}_prob_map'] = aux_out_dict['prob_map']
        if not self.training:
            out_dict['prob_map'] = \
                sum(list(map(out_dict.get, [f'{self.decode_head_name}_{i}_prob_map' for i in range(self.head_num)]))) / self.head_num
        return out_dict

import torch.nn as nn

from .base import BaseSegmentor
from .. import MSTfamily
from ..MSTfamily import SEGMENTORS
from ..ops import resize


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 head_weight=None):

        super(EncoderDecoder, self).__init__()
        self.backbone = MSTfamily.build_backbone(backbone)
        if neck is not None:
            self.neck = MSTfamily.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.run_mode = 'train'

        self.init_weights(pretrained=pretrained)

        self.head_weight = head_weight

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head_name = decode_head.type
        self.decode_head = MSTfamily.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.aux_head_name = 'Sequence_head'
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MSTfamily.build_head(head_cfg))
                    self.aux_head_name = f'{auxiliary_head}_{head_cfg.type}'
            else:
                self.aux_head_name = auxiliary_head.type
                self.auxiliary_head = MSTfamily.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        out = {'backbone_feat': x}
        return out

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward(self, img, img_metas=None):
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
        out_dict.update(decode_out_dict)
        # temp =  vis_img(petrificus_totalus(img))
        # plt_show(temp)

        if self.with_auxiliary_head:
            out_dict[f'{self.decode_head_name}_prob_map'] = out_dict['prob_map']
            out_dict.pop('prob_map')
            aux_out_dict = self.auxiliary_head(feat_out_dict['backbone_feat'])
            out_dict.update(aux_out_dict)
            out_dict[f'{self.aux_head_name}_prob_map'] = out_dict['prob_map']
            out_dict.pop('prob_map')
        if not self.training:
            if self.with_auxiliary_head:
                out_dict['prob_map'] = out_dict[f'{self.decode_head_name}_prob_map']
        return out_dict


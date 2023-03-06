type = 'EncoderDecoder'
backbone = dict(
    type='ResNetV1c',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    dilations=(1, 1, 1, 1),
    strides=(1, 2, 2, 2),
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=False,
    style='pytorch',
    contract_dilation=True)
decode_head = dict(
    type='UPerHead',
    in_channels=[256, 512, 1024, 2048],
    in_index=[0, 1, 2, 3],
    pool_scales=(1, 2, 3, 6),
    channels=512,
    dropout_ratio=0.1,
    num_classes=19,
    align_corners=False,
    norm_cfg=dict(type='BN',
                  requires_grad=True))

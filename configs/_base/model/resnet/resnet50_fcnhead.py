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
    type='FCNHead',
    in_channels=2048,
    in_index=3,
    channels=512,
    num_convs=2,
    concat_input=True,
    dropout_ratio=0.1,
    num_classes=19,
    norm_cfg=dict(type='BN', requires_grad=True),
    align_corners=False)

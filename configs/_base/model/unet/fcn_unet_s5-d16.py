type = 'EncoderDecoder'
backbone = dict(
    type='UNet',
    in_channels=3,
    base_channels=64,
    num_stages=5,
    strides=(1, 1, 1, 1, 1),
    enc_num_convs=(2, 2, 2, 2, 2),
    dec_num_convs=(2, 2, 2, 2),
    downsamples=(True, True, True, True),
    enc_dilations=(1, 1, 1, 1, 1),
    dec_dilations=(1, 1, 1, 1),
    with_cp=True,
    conv_cfg=None,
    norm_cfg=dict(type='BN', requires_grad=True),
    act_cfg=dict(type='ReLU'),
    upsample_cfg=dict(type='InterpConv'),
    norm_eval=False)
decode_head = dict(
    type='FCNHead',
    in_channels=64,
    in_index=4,
    channels=64,
    num_convs=1,
    concat_input=False,
    dropout_ratio=0.1,
    num_classes=2,
    norm_cfg=dict(type='BN', requires_grad=True),
    align_corners=False)

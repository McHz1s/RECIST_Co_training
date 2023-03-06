_Base_Config = 'configs/_base/model/hrnet/fcn_hr18.py'
backbone = dict(
    extra=dict(
        stage2=dict(num_channels=(48, 96)),
        stage3=dict(num_channels=(48, 96, 192)),
        stage4=dict(num_channels=(48, 96, 192, 384))))
decode_head = dict(
    in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384]))

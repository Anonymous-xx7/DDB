_base_ = ['deeplabv2red_r50-d8.py']
# models settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet101_v1c')))













'''###____pretty_text____###'''



'''
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet101_v1c'),
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DLV2Head',
        in_channels=2048,
        in_index=3,
        dilations=(6, 12),
        num_classes=19,
        align_corners=False,
        init_cfg=dict(
            type='Normal', std=0.01, override=dict(name='aspp_modules')),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
'''

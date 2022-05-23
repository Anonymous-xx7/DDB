_base_ = ['deeplabv2_r50-d8.py']
# Previous UDA methods only use the dilation rates 6 and 12 for DeepLabV2.
# This might be a bit hidden as it is caused by a return statement WITHIN
# a loop over the dilation rates:
# https://github.com/wasidennis/AdaptSegNet/blob/fca9ff0f09dab45d44bf6d26091377ac66607028/model/deeplab.py#L116
model = dict(decode_head=dict(dilations=(6, 12)))













'''###____pretty_text____###'''



'''
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'),
        depth=50,
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

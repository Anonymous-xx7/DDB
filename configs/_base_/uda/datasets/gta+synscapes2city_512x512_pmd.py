# dataset settings
source1_type = "GTADataset"
source1_root = "data/gta/"
source1_img_dir = "images"
source1_ann_dir = "labels"
source2_type = "SynscapesDataset"
source2_root = "data/synscapes/"
source2_img_dir = "img/rgb"
source2_ann_dir = "img/class"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
source1_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1280, 720)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
source2_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
cityscapes_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
source1_train = dict(
    type=source1_type,
    data_root=source1_root,
    img_dir=source1_img_dir,
    ann_dir=source1_ann_dir,
    pipeline=source1_train_pipeline,
)
source2_train = dict(
    type=source2_type,
    data_root=source2_root,
    img_dir=source2_img_dir,
    ann_dir=source2_ann_dir,
    pipeline=source2_train_pipeline,
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="STDataset",
        source=[source1_train, source2_train],
        target=dict(
            type="CityscapesDataset",
            data_root="data/cityscapes/",
            img_dir="leftImg8bit/train",
            ann_dir="gtFine/train",
            pipeline=cityscapes_train_pipeline,
        ),
        post_pmd=True,
        post_blur=True,
        img_norm_cfg=img_norm_cfg,
    ),
    val=dict(
        type="CityscapesDataset",
        data_root="data/cityscapes/",
        img_dir="leftImg8bit/val",
        ann_dir="gtFine/val",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CityscapesDataset",
        data_root="data/cityscapes/",
        img_dir="leftImg8bit/val",
        ann_dir="gtFine/val",
        pipeline=test_pipeline,
    ),
)










'''###____pretty_text____###'''



'''
source1_type = 'GTADataset'
source1_root = 'data/gta/'
source1_img_dir = 'images'
source1_ann_dir = 'labels'
source2_type = 'SynscapesDataset'
source2_root = 'data/synscapes/'
source2_img_dir = 'img/rgb'
source2_ann_dir = 'img/class'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
source1_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
source2_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
source1_train = dict(
    type='GTADataset',
    data_root='data/gta/',
    img_dir='images',
    ann_dir='labels',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1280, 720)),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
source2_train = dict(
    type='SynscapesDataset',
    data_root='data/synscapes/',
    img_dir='img/rgb',
    ann_dir='img/class',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1024, 512)),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='STDataset',
        source=[
            dict(
                type='GTADataset',
                data_root='data/gta/',
                img_dir='images',
                ann_dir='labels',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='Resize', img_scale=(1280, 720)),
                    dict(
                        type='RandomCrop',
                        crop_size=(512, 512),
                        cat_max_ratio=0.75),
                    dict(type='RandomFlip', prob=0.5),
                    dict(type='PhotoMetricDistortion'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ]),
            dict(
                type='SynscapesDataset',
                data_root='data/synscapes/',
                img_dir='img/rgb',
                ann_dir='img/class',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='Resize', img_scale=(1024, 512)),
                    dict(
                        type='RandomCrop',
                        crop_size=(512, 512),
                        cat_max_ratio=0.75),
                    dict(type='RandomFlip', prob=0.5),
                    dict(type='PhotoMetricDistortion'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ])
        ],
        target=dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=(1024, 512)),
                dict(type='RandomCrop', crop_size=(512, 512)),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]),
        post_pmd=True,
        post_blur=True,
        img_norm_cfg=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
'''

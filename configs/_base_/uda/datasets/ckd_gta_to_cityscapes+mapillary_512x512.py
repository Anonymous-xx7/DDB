source_type = "GTADataset"
source_root = "data/gta/"
source_img_dir = "images"
source_ann_dir = "labels"
target1_type = "CityscapesDatasetCustom"
target1_root = "data/cityscapes/"
target1_train_img_dir = "leftImg8bit/train"
target1_train_ann_dir = "gtFine/train"
target1_test_img_dir = "leftImg8bit/val"
target1_test_ann_dir = "gtFine/val"
target2_type = "MapillaryDataset"
target2_root = "data/mapillary/"
target2_train_img_dir = "training/images"
target2_test_img_dir = "half/val_img"
target2_train_ann_dir = "cityscape_trainIdLabel/train/label"
target2_test_ann_dir = "half/val_label"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
source_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1280, 720)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
target1_train_pipeline = [
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
target2_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1024, 512), keep_ratio=True, min_size=512),
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
target1_train = dict(
    type=target1_type,
    data_root=target1_root,
    img_dir=target1_train_img_dir,
    ann_dir=target1_train_ann_dir,
    pipeline=target1_train_pipeline,
)
target2_train = dict(
    type=target2_type,
    data_root=target2_root,
    img_dir=target2_train_img_dir,
    ann_dir=target2_train_ann_dir,
    pipeline=target2_train_pipeline,
)
target1_test = dict(
    type=target1_type,
    data_root=target1_root,
    img_dir=target1_test_img_dir,
    ann_dir=target1_test_ann_dir,
    pipeline=test_pipeline,
)
target2_test = dict(
    type=target2_type,
    data_root=target2_root,
    img_dir=target2_test_img_dir,
    ann_dir=target2_test_ann_dir,
    pipeline=test_pipeline,
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="STDataset",
        source=dict(
            type=source_type,
            data_root=source_root,
            img_dir=source_img_dir,
            ann_dir=source_ann_dir,
            pipeline=source_train_pipeline,
        ),
        target=[target1_train, target2_train],
        post_pmd=True,
        post_blur=True,
        mask="zero",
        img_norm_cfg=img_norm_cfg,
    ),
    val=[target1_test, target2_test],
    test=[target1_test, target2_test],
)




'''###____pretty_text____###'''



'''
source_type = 'GTADataset'
source_root = 'data/gta/'
source_img_dir = 'images'
source_ann_dir = 'labels'
target1_type = 'CityscapesDatasetCustom'
target1_root = 'data/cityscapes/'
target1_train_img_dir = 'leftImg8bit/train'
target1_train_ann_dir = 'gtFine/train'
target1_test_img_dir = 'leftImg8bit/val'
target1_test_ann_dir = 'gtFine/val'
target2_type = 'MapillaryDataset'
target2_root = 'data/mapillary/'
target2_train_img_dir = 'training/images'
target2_test_img_dir = 'half/val_img'
target2_train_ann_dir = 'cityscape_trainIdLabel/train/label'
target2_test_ann_dir = 'half/val_label'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
source_train_pipeline = [
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
target1_train_pipeline = [
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
target2_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True, min_size=512),
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
target1_train = dict(
    type='CityscapesDatasetCustom',
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
    ])
target2_train = dict(
    type='MapillaryDataset',
    data_root='data/mapillary/',
    img_dir='training/images',
    ann_dir='cityscape_trainIdLabel/train/label',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='Resize',
            img_scale=(1024, 512),
            keep_ratio=True,
            min_size=512),
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
    ])
target1_test = dict(
    type='CityscapesDatasetCustom',
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
    ])
target2_test = dict(
    type='MapillaryDataset',
    data_root='data/mapillary/',
    img_dir='half/val_img',
    ann_dir='half/val_label',
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
    ])
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='STDataset',
        source=dict(
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
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]),
        target=[
            dict(
                type='CityscapesDatasetCustom',
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
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ]),
            dict(
                type='MapillaryDataset',
                data_root='data/mapillary/',
                img_dir='training/images',
                ann_dir='cityscape_trainIdLabel/train/label',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        type='Resize',
                        img_scale=(1024, 512),
                        keep_ratio=True,
                        min_size=512),
                    dict(type='RandomCrop', crop_size=(512, 512)),
                    dict(type='RandomFlip', prob=0.5),
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
        post_pmd=True,
        post_blur=True,
        mask='zero',
        img_norm_cfg=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)),
    val=[
        dict(
            type='CityscapesDatasetCustom',
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
        dict(
            type='MapillaryDataset',
            data_root='data/mapillary/',
            img_dir='half/val_img',
            ann_dir='half/val_label',
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
            ])
    ],
    test=[
        dict(
            type='CityscapesDatasetCustom',
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
        dict(
            type='MapillaryDataset',
            data_root='data/mapillary/',
            img_dir='half/val_img',
            ann_dir='half/val_label',
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
            ])
    ])
'''

# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support UDA models

import warnings
from mmcv import Registry
from mmcv.cnn import MODELS as MMCV_MODELS

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
HEADS = MODELS
NECKS = MODELS
SEGMENTORS = MODELS
UDA = MODELS
MSDA = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_train_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.model.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.model.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    if 'uda' in cfg:
        cfg.uda.model = cfg.model
        cfg.uda.max_iters = cfg.runner.max_iters
        return UDA.build(
            cfg.uda, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    elif 'msda' in cfg:
        cfg.msda.model = cfg.model
        cfg.msda.max_iters = cfg.runner.max_iters
        return MSDA.build(
            cfg.msda, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    else:
        return SEGMENTORS.build(
            cfg.model, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

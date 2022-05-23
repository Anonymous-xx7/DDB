from mmseg.models.builder import LOSSES, build_loss
from .builder import (SEGMENTORS, BACKBONES, NECKS, HEADS, UDA, build_train_model, build_backbone, build_neck,
                      build_head, build_segmentor)
from .backbones import *
from .decode_heads import *
from .segmentors import *
from .uda import *

__all__ = [
    'BACKBONES', 'HEADS', 'NECKS', 'LOSSES', 'SEGMENTORS', 'UDA', 'build_backbone',
    'build_head', 'build_loss', 'build_train_model', 'build_segmentor', 'build_neck'
]

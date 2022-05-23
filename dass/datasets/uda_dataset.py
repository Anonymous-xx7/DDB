import numpy as np

from . import DATASETS


@DATASETS.register_module()
class UDADataset(object):
    def __init__(self, source, target, cfg):
        self.cfg = cfg
        self.source = source
        self.target = target
        self.ignore_index = target.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE
        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE

    def __getitem__(self, idx):
        s = self.source[idx % len(self.source)]
        target_idx = np.random.choice(range(len(self.target)))
        t = self.target[target_idx]
        return {**s, 'target_img_metas': t['img_metas'], 'target_img': t['img']}

    def __len__(self):
        return max(len(self.source), len(self.target))

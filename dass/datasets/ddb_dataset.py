import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

import random
import torch.nn as nn
import kornia
from dass.models.utils.dacs_transforms import denorm_, renorm_
from . import DATASETS


@DATASETS.register_module()
class DDBDataset(object):
    def __init__(self, source, target, cfg):
        self.cfg = cfg
        self.source = source
        self.target = target
        if hasattr(target, "ignore_index"):
            self.ignore_index = target.ignore_index
        else:
            self.ignore_index = source.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE
        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE

        self.post_pmd = cfg["post_pmd"]
        self.post_blur = cfg["post_blur"]
        self.img_mean = cfg.img_norm_cfg["mean"]
        self.img_std = cfg.img_norm_cfg["std"]
        self.bi_cu = cfg.bi_cu

    def __getitem__(self, idx):
        src = self.source[idx % len(self.source)]
        target_idx = np.random.choice(range(len(self.target)))
        tgt = self.target[target_idx]

        src_img, src_gt = src["img"].data, src["gt_semantic_seg"].data
        tgt_img, tgt_pl = (
            tgt["img"].data,
            torch.ones_like(src_gt, dtype=torch.long) * self.ignore_index,
        )
        cu_bridging = self.get_cu(src_img, tgt_img)
        if self.bi_cu:
            bi_cu_mask = cu_bridging["mask"].data * -1 + 1
            bi_cu_brg_img = bi_cu_mask * src_img + (1 - bi_cu_mask) * tgt_img
            bi_cu_brg = dict(
                img=DC(bi_cu_brg_img, stack=True), mask=DC(bi_cu_mask, stack=True)
            )
            ca_bridging = bi_cu_brg
        else:
            ca_bridging = self.get_ca(src_img, src_gt, tgt_img)
        return {
            **src,
            "target_img_metas": tgt["img_metas"],
            "target_img": tgt["img"],
            "cu_bridging": cu_bridging,
            "ca_bridging": ca_bridging,
        }

    def __len__(self):
        return max(len(self.source), len(self.target))

    def get_cu(self, src_img, tgt_img):
        cu_mask = self.get_cut_mask(src_img.shape)
        cu_brg_img = cu_mask * src_img + (1 - cu_mask) * tgt_img
        if self.post_pmd or self.post_blur:
            cu_brg_img = self.post_process(cu_brg_img)
        cu_brg = dict(img=DC(cu_brg_img, stack=True), mask=DC(cu_mask, stack=True))
        return cu_brg

    def get_ca(self, src_img, src_gt, tgt_img):
        ca_mask = self.get_class_mask(src_gt)
        ca_brg_img = ca_mask * src_img + (1 - ca_mask) * tgt_img
        if self.cfg.post_pmd or self.post_blur:
            ca_brg_img = self.post_process(ca_brg_img)
        ca_brg = dict(img=DC(ca_brg_img, stack=True), mask=DC(ca_mask, stack=True))
        return ca_brg

    def get_class_mask(self, s_gt):
        classes = torch.unique(s_gt)
        num_classes = classes.shape[0]
        class_choice = np.random.choice(
            num_classes, int((num_classes + num_classes % 2) / 2), replace=False
        )
        classes = classes[torch.Tensor(class_choice).long()]
        label, classes = torch.broadcast_tensors(
            s_gt, classes.unsqueeze(1).unsqueeze(2)
        )
        mask = label.eq(classes).sum(0, keepdims=True)
        return mask

    def get_cut_mask(self, img_shape):
        _, h, w = img_shape
        y_props = np.exp(
            np.random.uniform(low=0.0, high=1, size=(1,))
            * np.log(self.cfg.cu_mask_props)
        )
        x_props = self.cfg.cu_mask_props / y_props
        sizes = np.round(
            np.stack([y_props, x_props], axis=1) * np.array((h, w))[None, :]
        )

        positions = np.round(
            (np.array((h, w)) - sizes)
            * np.random.uniform(low=0.0, high=1.0, size=sizes.shape)
        )
        rectangle = np.append(positions, positions + sizes, axis=1)

        mask = torch.zeros((1, h, w)).long()
        y0, x0, y1, x1 = rectangle[0]
        mask[0, int(y0) : int(y1), int(x0) : int(x1)] = 1
        return mask

    @staticmethod
    def color_jitter(data, mean, std, s=0.2, p=0.2):
        # s is the strength of colorjitter
        if random.uniform(0, 1) > p:
            if isinstance(s, dict):
                seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
            else:
                seq = nn.Sequential(
                    kornia.augmentation.ColorJitter(
                        brightness=s, contrast=s, saturation=s, hue=s
                    )
                )
            denorm_(data, mean, std)
            data = seq(data).squeeze(0)
            renorm_(data, mean, std)
        return data

    @staticmethod
    def gaussian_blur(data=None, p=0.5):
        if random.uniform(0, 1) > p:
            data.unsqueeze_(0)
            sigma = np.random.uniform(0.15, 1.15)
            kernel_size_y = int(
                np.floor(
                    np.ceil(0.1 * data.shape[1])
                    - 0.5
                    + np.ceil(0.1 * data.shape[1]) % 2
                )
            )
            kernel_size_x = int(
                np.floor(
                    np.ceil(0.1 * data.shape[2])
                    - 0.5
                    + np.ceil(0.1 * data.shape[2]) % 2
                )
            )
            kernel_size = (kernel_size_y, kernel_size_x)
            seq = nn.Sequential(
                kornia.filters.GaussianBlur2d(
                    kernel_size=kernel_size, sigma=(sigma, sigma)
                )
            )
            data = seq(data).squeeze(0)
        return data

    def post_process(self, img):
        mean = torch.as_tensor(self.img_mean).view(3, 1, 1)
        std = torch.as_tensor(self.img_std).view(3, 1, 1)
        if self.post_pmd:
            img = self.color_jitter(img, mean, std)
        if self.post_blur:
            img = self.gaussian_blur(img)
        return img

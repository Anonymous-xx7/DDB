from copy import deepcopy
from mmseg.models.builder import SEGMENTORS
from mmcv.runner import BaseModule
from mmcv.runner import load_state_dict, _load_checkpoint
from mmcv.utils.logging import get_logger, logger_initialized, print_log
from .base import BaseSegmentor
import torch


@SEGMENTORS.register_module()
class Tester(BaseSegmentor):
    def __init__(
        self,
        segmentor,
        weight,
        **kwargs,
    ):
        """
        This model is only used for testing
        """
        super().__init__(None)

        segmentor = deepcopy(segmentor)
        from dass.models.builder import build_segmentor

        self.segmentor = build_segmentor(segmentor)
        self.weight = weight

    def init_weights(self):
        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else "mmcv"
        checkpoint = _load_checkpoint(
            self.weight, map_location="cpu", logger=logger_name
        )
        print_log(f"use checkpoint {self.weight}", logger=logger_name)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        prefixs = ("model", "stu_model")
        for prefix in prefixs:
            if not prefix.endswith("."):
                prefix += "."
            prefix_len = len(prefix)
            new_state_dict = {
                k[prefix_len:]: v for k, v in state_dict.items() if k.startswith(prefix)
            }
            if len(new_state_dict) > 0:
                state_dict = new_state_dict
                print_log(f"get state dict with prefix {prefix}", logger=logger_name)
                break
        load_state_dict(self.segmentor, state_dict, strict=True, logger=logger_name)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def inference(self, img, img_meta, rescale):
        return self.segmentor.inference(img, img_meta, rescale)

    def forward_train(self, *args, **kwargs):
        return self.segmentor.forward_train(args, kwargs)

    def encode_decode(self, *args, **kwargs):
        return self.segmentor.encode_decode(args, kwargs)

    def extract_feat(self, *args, **kwargs):
        return self.segmentor.extract_feat(args, kwargs)

import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmcv.utils import print_log
from mmseg.core import add_prefix
from .uda_decorator import UDADecorator, get_module
from ..builder import build_segmentor, UDA
from ..utils import denorm, get_mean_std
from ..utils import subplotimg
from torch.nn.parallel.distributed import _find_tensors

mpl.use("Agg")  # To solve the main thread is not in main loop error


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(), model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            return False
    return True


@UDA.register_module()
class ST(UDADecorator):
    def __init__(self, **cfg):
        super(ST, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg["max_iters"]
        self.alpha = cfg["alpha"]
        self.pseudo_threshold = cfg["pseudo_threshold"]
        self.debug_img_interval = cfg["debug_img_interval"]

        ema_cfg = deepcopy(cfg["model"])
        self.ema_model = build_segmentor(ema_cfg)

        self.distilled_model_path = cfg["distilled_model_path"]
        self.proto_path = cfg["proto_path"]
        self.proto_rectify = cfg["proto_rectify"]
        self.moving_proto = cfg["moving_proto"]
        self.proto_momentum = cfg["proto_momentum"]
        self.is_mixup = cfg.get("is_mixup", False)
        self.vectors = torch.zeros([self.num_classes, 256])
        self.vectors_num = torch.zeros([self.num_classes])

    def calculate_mean_vector(self, feature, logit, thresh=None):
        outputs_softmax = F.softmax(logit, dim=1)
        if thresh is None:
            thresh = -1
        conf = outputs_softmax.max(dim=1, keepdim=True)[0]
        mask = conf.ge(thresh)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        outputs_pred = outputs_argmax

        scale_factor = F.adaptive_avg_pool2d(outputs_pred * mask, 1)
        vectors = []
        ids = []
        for n in range(feature.size()[0]):
            for t in range(self.num_classes):
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feature[n] * outputs_pred[n][t] * mask[n]
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def update_objective_vector(
        self, idx, vector, mode="moving_average", start_mean=True
    ):
        if vector.sum().item() == 0:
            return
        if start_mean and self.vectors_num[idx].item() < 100:
            mode = "mean"
        if mode == "moving_average":
            self.vectors[idx] = (
                self.vectors[idx] * self.proto_momentum
                + (1 - self.proto_momentum) * vector.squeeze()
            )
            self.vectors_num[idx] += 1
            self.vectors_num[idx] = min(self.vectors_num[idx], 3000)
        elif mode == "mean":
            self.vectors[idx] = (
                self.vectors[idx] * self.vectors_num[idx] + vector.squeeze()
            )
            self.vectors_num[idx] += 1
            self.vectors[idx] = self.vectors[idx] / self.vectors_num[idx]
            self.vectors_num[idx] = min(self.vectors_num[idx], 3000)
            pass
        else:
            raise NotImplementedError(
                "no such updating way of objective vectors {}".format(mode)
            )

    def process_label(self, label):
        batch, _, w, h = label.size()
        pred1 = torch.zeros(batch, self.num_classes + 1, w, h).cuda()
        idx = torch.where(
            label < self.num_classes, label, torch.Tensor([self.num_classes]).cuda()
        )
        pred1 = pred1.scatter_(1, idx.long(), 1)
        return pred1

    def get_prototype_weight(self, feature: torch.Tensor):
        feat_proto_dis = self.feat_proto_distance(feature)
        feat_nearest_proto_dis, _ = feat_proto_dis.min(dim=1, keepdim=True)
        feat_proto_dis = feat_proto_dis - feat_nearest_proto_dis
        weight = F.softmax(-feat_proto_dis, dim=1)
        return weight

    def feat_proto_distance(self, feature: torch.Tensor):
        n, _, h, w = feature.shape
        feat_proto_distance = -torch.ones((n, self.num_classes, h, w)).cuda()
        for i in range(self.num_classes):
            feat_proto_distance[:, i, :, :] = torch.norm(
                self.vectors[i].reshape(-1, 1, 1).expand(-1, h, w) - feature,
                2,
                dim=1,
            )
        return feat_proto_distance

    def get_model(self, name="model"):
        if "model" not in name:
            name = "_".join([name, "model"])
        return get_module(getattr(self, name))

    def _init_distill_weights(self):
        distill_ckpt = torch.load(self.distilled_model_path, map_location="cpu")
        self._load_checkpoint("model", distill_ckpt)
        print_log(f"Load checkpoint from {self.distilled_model_path}", "mmseg")
        if self.proto_rectify:
            self.vectors = torch.load(self.proto_path, map_location="cpu").cuda()

    def _load_checkpoint(self, name, checkpoint):
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "stu" in k:
                new_state_dict[k.replace("stu_model.", "")] = v
        self.get_model(name).load_state_dict(new_state_dict, strict=True)

    def _init_ema_weights(self):
        for param in self.get_model("ema").parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_model("ema").parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(
            self.get_model("ema").parameters(), self.get_model().parameters()
        ):
            if not param.data.shape:  # scalar tensor
                ema_param.data = (
                    alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
                )
            else:
                ema_param.data[:] = (
                    alpha_teacher * ema_param[:].data[:]
                    + (1 - alpha_teacher) * param[:].data[:]
                )

    def train_step(self, data_batch, optimizer, ddp_reducer=None, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        self.ddp_reducer = ddp_reducer  # store ddp reducer
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop("loss", None)  # remove the unnecessary 'loss'
        outputs = dict(log_vars=log_vars, num_samples=len(data_batch["img_metas"]))
        self.ddp_reducer = None  # drop ddp reducer
        return outputs

    def forward_train(
        self, img, img_metas, gt_semantic_seg, target_img, target_img_metas, bridging
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        if self.is_mixup:
            alpha = 2
            lam = np.random.beta(alpha, alpha)
            lam = torch.tensor(lam, device=dev)
            bridging["img"] = lam * img + (1 - lam) * target_img
        # Init/update ema model
        if self.local_iter == 0:
            if self.distilled_model_path is not None:
                self._init_distill_weights()
            self._init_ema_weights()
            assert _params_equal(self.get_model("ema"), self.get_model())
        else:
            self._update_ema(self.local_iter)

        means, stds = get_mean_std(img_metas, dev)

        # Train on source images
        clean_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg)
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        if self.is_mixup:
            clean_loss = clean_loss * lam
        log_vars.update(clean_log_vars)
        if getattr(self, "ddp_reducer", None):
            self.ddp_reducer.prepare_for_backward(_find_tensors(clean_loss))
        clean_loss.backward()

        # Generate pseudo-label
        for m in self.get_model("ema").modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        if self.proto_rectify:
            ema_logits, ema_feature = self.get_model("ema").encode_decode(
                target_img, target_img_metas, return_feature=True
            )
            if self.moving_proto:
                ema_vectors, ema_ids = self.calculate_mean_vector(
                    ema_feature, ema_logits
                )
                for t in range(len(ema_ids)):
                    self.update_objective_vector(
                        ema_ids[t], ema_vectors[t].detach(), start_mean=False
                    )
            ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
            pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
            proto_weights = self.get_prototype_weight(ema_feature)
            ema_softmax = proto_weights * ema_softmax
            rectify_pseudo_prob, rectify_pseudo_label = torch.max(ema_softmax, dim=1)
            ps_large_p = rectify_pseudo_prob.ge(self.pseudo_threshold).long() == 1
        else:
            ema_logits = self.get_model("ema").encode_decode(
                target_img, target_img_metas
            )
            ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
            pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1

        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=dev)
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        # Apply mixing
        mixed_img, mix_mask = bridging["img"], bridging["mask"]
        if self.proto_rectify:
            mixed_lbl = mix_mask * gt_semantic_seg + (
                1 - mix_mask
            ) * rectify_pseudo_label.unsqueeze(1)
        else:
            mixed_lbl = mix_mask * gt_semantic_seg + (
                1 - mix_mask
            ) * pseudo_label.unsqueeze(1)
        mixed_weight = (
            mix_mask.squeeze(1) * gt_pixel_weight
            + (1 - mix_mask.squeeze(1)) * pseudo_weight
        )

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, mixed_weight
        )
        mix_losses = add_prefix(mix_losses, "mix")
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        if self.is_mixup:
            mix_loss = mix_loss * (1 - lam)
        log_vars.update(mix_log_vars)
        if getattr(self, "ddp_reducer", None):
            self.ddp_reducer.prepare_for_backward(_find_tensors(mix_loss))
        mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg["work_dir"], "visual_debug")
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 4
                if self.proto_rectify:
                    cols += 1
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        "hspace": 0.1,
                        "wspace": 0,
                        "top": 0.95,
                        "bottom": 0,
                        "right": 1,
                        "left": 0,
                    },
                )
                subplotimg(axs[0][0], vis_img[j], "Source Image")
                subplotimg(axs[1][0], vis_trg_img[j], "Target Image")
                subplotimg(
                    axs[0][1], gt_semantic_seg[j], "Source Seg GT", cmap="cityscapes"
                )
                subplotimg(
                    axs[1][1], pseudo_label[j], "Target Seg PL", cmap="cityscapes"
                )
                subplotimg(axs[0][2], vis_mixed_img[j], "Mixed. Image")
                subplotimg(axs[1][2], mix_mask[j][0], "Domain Mask", cmap="gray")
                subplotimg(axs[0][3], mixed_weight[j], "Pseudo W.", vmin=0, vmax=1)
                subplotimg(axs[1][3], mixed_lbl[j], "Mixed PL", cmap="cityscapes")
                if self.proto_rectify:
                    subplotimg(
                        axs[1][4],
                        rectify_pseudo_label[j],
                        "Target Rec. Seg PL",
                        cmap="cityscapes",
                    )
                for ax in axs.flat:
                    ax.axis("off")
                plt.savefig(
                    os.path.join(out_dir, f"{(self.local_iter + 1):06d}_{j}.png")
                )
                plt.close()
        self.local_iter += 1

        return log_vars

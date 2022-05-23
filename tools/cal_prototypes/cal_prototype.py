import argparse
import os.path as osp

import mmcv
import torch
from dass.apis import set_random_seed
from dass.datasets import build_dataloader, build_dataset
from dass.models import build_train_model
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import Config
from mmseg.utils import get_root_logger, setup_multi_processes

from class_features import ClassFeatures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate the prototype for trained model"
    )
    parser.add_argument("config", help="trained model config file path")
    parser.add_argument("checkpoint", help="checkpoint file path")
    parser.add_argument("save_dir", type=str, default=".")
    parser.add_argument("--postfix", default=None, help="postfix for saved file name")
    parser.add_argument(
        "--epochs", default=4, type=int, help="epochs for calculating the prototypes"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use " "(only applicable to non-distributed training)",
    )
    return parser.parse_args()


def calculate_prototypes(model, data_loader, logger, epochs):
    dev = torch.device(model.output_device)
    loader_indices = data_loader.batch_sampler
    class_features = ClassFeatures(numbers=model.module.num_classes, dev=dev)

    for epoch in range(epochs):
        logger.info(f"Calculating the prototypes on Epoch {epoch + 1}..")
        prog_bar = mmcv.ProgressBar(len(dataset))
        for batch_indices, data in zip(loader_indices, data_loader):
            del data["gt_semantic_seg"]
            data["img"] = data["img"].data[0].to(dev)
            data["img_metas"] = data["img_metas"].data[0]

            with torch.no_grad():
                logits, features = model.module.get_model().encode_decode(
                    return_feature=True, **data
                )
                vectors, ids = class_features.calculate_mean_vector(features, logits)
                for t in range(len(ids)):
                    class_features.update_objective_SingleVector(
                        ids[t], vectors[t].detach(), "mean"
                    )
            batch_size = data_loader.batch_size
            for _ in range(batch_size):
                prog_bar.update()

    return class_features.objective_vectors


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    setup_multi_processes(cfg)
    # set random seed and cudnn_benchmark
    set_random_seed(cfg.seed, deterministic=True)
    save_pre_fix = osp.splitext(osp.basename(args.config))[0]
    save_path = args.save_dir
    save_path = osp.join(save_path, "prototypes")
    mmcv.mkdir_or_exist(osp.abspath(save_path))
    if args.postfix:
        postfix = args.postfix + ".pth"
    else:
        postfix = ".pth"
    save_path = osp.join(save_path, save_pre_fix + postfix)

    log_file = osp.join(args.save_dir, f"run_for_{save_pre_fix}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(f"RUNDIR: {args.save_dir}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    model = build_train_model(
        cfg, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()

    dataset = build_dataset(cfg.data.train.target)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        seed=cfg.seed,
        drop_last=False,
    )

    prototypes = calculate_prototypes(model, data_loader, logger, args.epochs)
    logger.info(f"Save the prototypes to {save_path}")
    torch.save(prototypes, save_path)

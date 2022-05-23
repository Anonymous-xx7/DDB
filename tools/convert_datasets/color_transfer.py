import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import mmcv
import argparse
import os.path as osp


def color_transfer(src_img: np.ndarray, ref_img: np.ndarray):
    src_lab = rgb2lab(src_img)
    ref_lab = rgb2lab(ref_img)
    src_mean, src_std = src_lab.mean(axis=(0, 1), keepdims=True), src_lab.std(
        axis=(0, 1), keepdims=True
    )
    ref_mean, ref_std = ref_lab.mean(axis=(0, 1), keepdims=True), ref_lab.std(
        axis=(0, 1), keepdims=True
    )
    output_lab = (src_lab - src_mean) / src_std * ref_std + ref_mean
    return lab2rgb(output_lab)


def color_transfer_save(src_path, ref_path, dst_path):
    src = Image.open(src_path)
    src = np.asarray(src)
    ref = Image.open(ref_path)
    ref = np.asarray(ref)
    dst = color_transfer(src, ref)
    dst = Image.fromarray(np.uint8(np.clip(dst * 255, 0, 255)))
    dst.save(dst_path)


def progress(task):
    color_transfer_save(*task)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Synscapes annotations to TrainIds and Color"
    )
    parser.add_argument("src_dir")
    parser.add_argument("ref_dir")
    parser.add_argument("dst_dir")
    parser.add_argument("--nproc", default=4, type=int, help="number of process")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.dst_dir)
    src_paths = list(
        osp.join(args.src_dir, p)
        for p in mmcv.scandir(args.src_dir, suffix=".png", recursive=True)
    )
    ref_paths = list(
        osp.join(args.ref_dir, p)
        for p in mmcv.scandir(args.ref_dir, suffix=".png", recursive=True)
    )
    ref_paths = [ref_paths[np.random.randint(len(ref_paths))] for _ in src_paths]
    dst_paths = [
        osp.join(args.dst_dir, osp.basename(filepath)) for filepath in src_paths
    ]
    if args.nproc > 1:
        mmcv.track_parallel_progress(
            progress, list(zip(src_paths, ref_paths, dst_paths)), args.nproc
        )
    else:
        mmcv.track_progress(progress, list(zip(src_paths, ref_paths, dst_paths)))


if __name__ == "__main__":
    main()

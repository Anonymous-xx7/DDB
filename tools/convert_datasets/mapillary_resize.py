import argparse
import json
import os.path as osp

import cv2
import mmcv
import numpy as np
from PIL import Image


def resize_half(args):
    (
        id,
        image_folder,
        label_folder,
        dst_image_folder,
        dst_label_folder,
    ) = args
    im_file = osp.join(image_folder, f"{id}.jpg")
    label_file = osp.join(label_folder, f"{id}.png")
    dst_im_file = osp.join(dst_image_folder, f"{id}.jpg")
    dst_label_file = osp.join(dst_label_folder, f"{id}.png")
    im = Image.open(im_file)
    h, w = im.size
    im = im.resize([h // 2, w // 2], resample=Image.BICUBIC)
    im.save(dst_im_file)
    label = Image.open(label_file)
    label = label.resize([h // 2, w // 2], resample=Image.NEAREST)
    label.save(dst_label_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder")
    parser.add_argument("label_folder")
    parser.add_argument("dst_image_folder")
    parser.add_argument("dst_label_folder")
    parser.add_argument("--nproc", default=4, type=int, help="number of process")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.dst_image_folder)
    mmcv.mkdir_or_exist(args.dst_label_folder)
    imgs = []
    for filename in mmcv.scandir(args.image_folder, suffix=".jpg"):
        id = osp.splitext(filename)[0]
        imgs.append(id)
    tasks = [
        (
            id,
            args.image_folder,
            args.label_folder,
            args.dst_image_folder,
            args.dst_label_folder,
        )
        for id in imgs
    ]
    if args.nproc > 1:
        mmcv.track_parallel_progress(resize_half, tasks, args.nproc)
    else:
        mmcv.track_progress(resize_half, tasks)


if __name__ == "__main__":
    main()

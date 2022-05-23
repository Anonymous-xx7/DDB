import argparse
import os
import os.path as osp
import numpy as np
import mmcv
from PIL import Image

CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

PALETTE = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--nproc", default=8, type=int, help="number of process")
    args = parser.parse_args()
    return args


def segmap2colormap(segmap: np.ndarray) -> np.ndarray:
    if segmap.ndim == 3:
        segmap = segmap.squeeze(0)
    H, W = segmap.shape
    colormap = np.zeros([H, W, 3], dtype=np.uint8)
    for trainId, color in enumerate(PALETTE):
        colormap[segmap[:, :] == trainId] = np.array(
            color, dtype=np.uint8
        )
    return colormap


def task(info):
    src_path, dst_path = info
    label = Image.open(src_path)
    label = np.array(label)
    color = segmap2colormap(label)
    color = Image.fromarray(color)
    color.save(dst_path)


def main():
    args = parse_args()
    files = mmcv.scandir(args.src, suffix=".png", recursive=True)
    infos = [(osp.join(args.src, f), osp.join(args.dst, f)) for f in files]
    mmcv.mkdir_or_exist(args.dst)
    if args.nproc > 1:
        mmcv.track_parallel_progress(task, infos, args.nproc)
    else:
        mmcv.track_progress(task, infos)
if __name__=='__main__':
    main()
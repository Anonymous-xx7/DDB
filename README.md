## Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation

## Overview 
This repo is a PyTorch implementation of applying DDB (Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation) to semantic segmentation. The code is based on mmsegmentaion.

More details can be found in Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation.

## Enviroment
In this project, we use python 3.8.13 and pytorch==1.8.1, torchvision==0.9.1, mmcv-full==1.4.7.

If your device has internet access, you could set up as follows:

```shell
conda create -n dass python=3.8
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
## Results
config | train dataset|validation dataset | mIoU 
---------|----------|--------|-------
weights/gta+syn2cs/r2-ckd-pro-bs1x4/weight.pth |gta| cityscape | 62.71 
weights/gta+syn2cs/r2-ckd-pro-bs1x4/weight.pth |gta+syn| cityscape | 68.99 
weights/gta2cs+map/s2-ckd-pro-bs1x4/weight.pth |gta| cityscape | 60.38 
weights/gta2cs+map/s2-ckd-pro-bs1x4/weight.pth |gta| mapillary | 56.85
The above weight and log can be obtained through [BaiduYun](https://pan.baidu.com/s/1uqSItKTTB3eCCj9sFT1Tww?pwd=60il). After downloading, please put it under the project folder
## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia:** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.  
**mapillary** Please, download MAPILLARY v1.2 from https://research.mapillary.com/  
Then, you should prepare data as follows:
```shell
# All data are listed in /mnt/lustreold/share_data/chenlin/data
cd DASS
ln -s /mnt/lustreold/share_data/chenlin/data data
# If you prepare the data at the first time, you should convert the data for training and validation
python tools/convert_datasets/gta.py data/gta/ # Source domain
python tools/convert_datasets/synthia.py data/synthia/ # Source domain
python tools/convert_datasets/synscapes.py data/synscapes/ # Source domain
# convert mapillary to cityscape format and resize it for efficient validation
python tools/convert_datasets/mapillary2cityscape.py data/mapillary/ \
data/mapillary/cityscape_trainIdLabel --train_id # Source domain
python tools/convert_datasets/mapillary_resize.py data/mapillary/validation/images data/mapillary/ \ cityscape_trainIdLabel/val/label data/mapillary/half/val_img data/mapillary/half/val_label
```

The final folder structure should look like this:

```none
DASS
├── ...
├── weights
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synscapes
│   │   ├── img
│   │   │   ├── class
│   │   │   ├── rgb
│   ├── mapillary
│   │   ├── training
│   │   ├── cityscape_trainIdLabel
│   │   ├── half
│   │   |   ├── val_img
│   │   |   ├── val_label
├── ...
```

## Evaluation
Download the folder  [weights](https://pan.baidu.com/s/1uqSItKTTB3eCCj9sFT1Tww?pwd=60il) and place it in the project directory
Verify by selecting the different config files in `configs/tests`
```shell
python tools/test.py {config} inconfig --eval mIoU
```



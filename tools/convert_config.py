# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

from mmcv import Config

FLAG = "____pretty_text____"


def convert(path: str) -> bool:
    try:
        text = Config.fromfile(path).pretty_text
    except:
        print(f"convert {path} False")
        text = "False to Parse"
    with open(path, "r", encoding="utf-8") as f:
        content = []
        for line in f:
            if line.find(FLAG) == -1:
                content.append(line)
            else:
                break
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(content)
        f.writelines(["\n"])
        f.write(f"'''###{FLAG}###'''")
        f.writelines(["\n"] * 4)
        f.write("'''\n")
        f.write(text)
        f.write("'''\n")
    print(f"convert {path} True")
    return True


from multiprocessing import Pool
import os, time, random


def main():
    start = time.time()
    config_dir = "/home/wzx/DASS/DASS/configs"
    configs = []
    for root, dirs, files in os.walk(config_dir):
        for file in files:
            configs.append(osp.join(root, file))
    p = Pool(32)
    for config in configs:
        p.apply_async(convert, args=(config,))
    p.close()
    p.join()
    end = time.time()
    print(f"using {end-start} s")


if __name__ == "__main__":
    main()

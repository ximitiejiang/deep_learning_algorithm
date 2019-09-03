#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:00:51 2019

@author: ubuntu
"""
# %% part1: run below shell script
"""
# -----install addict-----
pip3 install addict

# -----create data symlink-----
cd simple_ssd_pytorch
mkdir -p ./data
cd data
ln -s /home/ubuntu/MyDatasets/coco                             # change to your own data directory
ln -s /home/ubuntu/MyDatasets/voc/VOCdevkit                    # change to your own data directory

# -----creat work dir------
cd ..
mkdir -p ./work_dirs

# -----create weights symlink-----
mkdir -p ./weights
cd weights
ln -s /media/ubuntu/4430C54630C53FA2/SuLiang/MyWeights/myssd    # change to your own weights directory

"""
# %% part2: add sys path
import sys, os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)
    
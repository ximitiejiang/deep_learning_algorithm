#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:00:51 2019

@author: ubuntu
"""
# %% part1 dependency
"""
# -----install mmcv-----
$ pip3 install mmcv . #

# -----install addict-----
$ pip3 install addict

# -----create data symlink-----
$ cd ssd
$ mkdir data
$ ln -s /home/ubuntu/MyDatasets/coco
$ ln -s /home/ubuntu/MyDatasets/voc/VOCdevkit

# -----create weights symlink-----
$ cd ssd
$ mkdir weights
$ ln -s 
"""
# %% part2
# add sys path
import sys, os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)
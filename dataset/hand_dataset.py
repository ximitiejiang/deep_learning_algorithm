#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:02:04 2019

@author: ubuntu
"""
import numpy as np
import cv2
from dataset.base_dataset import BasePytorchDataset


class HandDataset(BasePytorchDataset):
    """牛津大学手部检测数据集：一个238M的小型数据集，可用来做检测，
    参考：http://www.robots.ox.ac.uk/~vgg/data/hands/
    """
    def __init__(self, root_path='/home/ubuntu/MyDatasets/hand_dataset/', ):
        pass
    
    def __getitem__(self, idx):
        pass
    
    def __len__(self):
        pass
    
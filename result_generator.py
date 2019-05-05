#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 07:09:05 2019

@author: ubuntu
"""

from model.one_stage_detector import OneStageDetector

import sys,os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)
    
from utils.tester import TestImgResultGenerator

def main():
    model_class = OneStageDetector
    dataset_name = 'traffic_sign'                           # 修改数据集类
    img_root = './data/traffic_sign/Test_fix/'             # 修改目标图片路径
    # support jpg/png list...
    img_names = os.listdir(img_root)  
    img_paths = [os.path.join(img_root, name) for name in img_names]  
    
    config_file = './config/cfg_retinanet_r50_fpn_trafficsign.py'
    weights_path = './weights/mytrafficsign/retinanet/epoch_20.pth'
    writetofile = './data/traffic_sign/submit_retinanet_0505.csv'
           
    test_img = TestImgResultGenerator(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
    # 不显示结果图片，不保存结果图片，写入结果数据
    test_img.run([img_paths[0]], show=False, save=None, writeto=writetofile)


if __name__=='__main__':
    main()

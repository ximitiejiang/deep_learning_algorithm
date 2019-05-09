#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 07:09:05 2019

@author: ubuntu
"""

from model.one_stage_detector import OneStageDetector
import csv
import sys,os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)
    
from utils.tester import TestImgResultGenerator


def result_summary(result_file):
    all_types = []
    with open(result_file) as f:
        lines = csv.reader(f)
        for line in lines:
            all_types.append(int(line[-1]))
    
        

def main():
    model_class = OneStageDetector
    dataset_name = 'traffic_sign'                           
#    img_root = './data/traffic_sign/_Test_fix/'
    img_root = '/media/ubuntu/4430C54630C53FA2/SuLiang/MyDatasets/traffic_sign/Test_fix/'             # 修改目标图片路径
    # support jpg/png list...
    img_names = os.listdir(img_root)  
    img_paths = [os.path.join(img_root, name) for name in img_names]  
    
#    config_file = './config/cfg_retinanet_r50_fpn_trafficsign.py'             # 修改配置文件
#    weights_path = './weights/mytrafficsign/retinanet_resnet/epoch_20.pth'    # 修改模型参数
#    writetofile = './data/traffic_sign/submit_retinanet_0506.csv'             # 修改输出文件名
    
    config_file = './config/cfg_retinanet_x101_64x4d_fpn_trafficsign.py'      # 修改配置文件
    weights_path = './weights/mytrafficsign/retinanet_resnext/epoch_15.pth'    # 修改模型参数
    writetofile = './data/traffic_sign/submit_retinanet_0508.csv'             # 修改输出文件名
           
    test_img = TestImgResultGenerator(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
    # 不显示结果图片，不保存结果图片，写入结果数据
    test_img.run(img_paths, show=False, save=None, writeto=writetofile)


if __name__=='__main__':
    main()

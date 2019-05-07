#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:47:41 2019

@author: suliang
"""
from dataset.utils import vis_bbox
import matplotlib.pyplot as plt
from dataset.trafficsign_dataset import TrafficSign

#    """data comes from DataFoutain contest: 基于虚拟仿真环境下的自动驾驶交通标志识别-2019/4
#    refer to: https://www.datafountain.cn/competitions/339/datasets
#    数据集ann_file说明：
#        ['filename', x1,y1,x2,y2,x3,y3,x4,y4,type]
#    类别说明：总共21类
#    {0:其他, 1:停车场, 2:停车让行, 3:右侧行驶, 4: 左转右转, 5:大客车通行, 
#    6:左侧行驶, 7:慢行, 8:机动车直行右转, 9:注意行人, 10:环岛形式,
#    11:直行右转, 12:禁止大客车, 13:禁止摩托车, 14:禁止机动车, 15:禁止非机动车,
#    16:禁止鸣喇叭, 17:立交直行转弯, 18:限速40公里, 19:限速30公里, 20:鸣喇叭}
#    """
#    
##    CLASSES = ('other', 'parking_lot', 'stop', 'right', 'left-right', 'bus', 
#               'left', 'slow', 'car-forard-right', 'person', 'island', 
#               'forward-right', 'bus-forbidden', 'motor-forbidden', 'car-forbidden', 'non-car-forbidden', 
#               'horn-forbidden', 'cross-forward-turn','speed40', 'speed30', 'horn')
#    CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
#            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')


if __name__ == "__main__":
    """用于检查数据集是否正确，运行之前需要把数据集中__getitem__()函数中return prepare_data()换成_prepare_data()"""    
    mode = 'train'
    if mode=='train':  # 训练数据集19,999
        ann_file = "./data/traffic_sign/train_label_fix.csv"
        img_prefix = "./data/traffic_sign/Train_fix"
        img_scale = (1333,800)
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], 
                            std=[58.395, 57.12, 57.375], 
                            to_rgb=True)
        extra_aug = dict(req_sizes = [1333,800])
#        extra_aug = None
        
        dataset = TrafficSign(ann_file, img_prefix, img_scale, img_norm_cfg, extra_aug=None)
        types, [a,b,c,d,e,f] = dataset.summarize(show=True)
        
        class_names = dataset.CLASSES  # class name (string) from '0' to '20'
        data = dataset[8455]   
        # [198,1222, 1683, 2683, 11164,] bbox位置不对，
        # [18284,11464,] 看不到有标签，
        print("h=%d, w=%d"% (data['img_meta']['ori_shape'][0], data['img_meta']['ori_shape'][1]))
        vis_bbox(data['img'], data['gt_bboxes'], labels=data['gt_labels'], class_names=class_names, 
                 instance_colors=None, alpha=1., linewidth=1.5, ax=None, saveto=None)   # img should be bgr/hwc
    

    

    
    
    
    
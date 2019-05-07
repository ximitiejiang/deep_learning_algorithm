#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:47:41 2019

@author: suliang
"""
import numpy as np
import torch
import cv2
import os
import csv
from torch.utils.data import Dataset
from dataset.utils import vis_bbox
import matplotlib.pyplot as plt
from dataset.transforms import (ImageTransform, BboxTransform, Numpy2Tensor) 
from dataset.random_fscrop_transform import RandomFSCropTransform
from model.parallel.data_container import DataContainer as DC

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))

def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.
    从[(1333,800)]中随机选择一个比例
    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale

class TrafficSign(Dataset):
    """data comes from DataFoutain contest: 基于虚拟仿真环境下的自动驾驶交通标志识别-2019/4
    refer to: https://www.datafountain.cn/competitions/339/datasets
    数据集ann_file说明：
        ['filename', x1,y1,x2,y2,x3,y3,x4,y4,type]
    类别说明：总共21类
    {0:其他, 1:停车场, 2:停车让行, 3:右侧行驶, 4: 左转右转, 5:大客车通行, 
    6:左侧行驶, 7:慢行, 8:机动车直行右转, 9:注意行人, 10:环岛形式,
    11:直行右转, 12:禁止大客车, 13:禁止摩托车, 14:禁止机动车, 15:禁止非机动车,
    16:禁止鸣喇叭, 17:立交直行转弯, 18:限速40公里, 19:限速30公里, 20:鸣喇叭}
    
    """
    
#    CLASSES = ('other', 'parking_lot', 'stop', 'right', 'left-right', 'bus', 
#               'left', 'slow', 'car-forard-right', 'person', 'island', 
#               'forward-right', 'bus-forbidden', 'motor-forbidden', 'car-forbidden', 'non-car-forbidden', 
#               'horn-forbidden', 'cross-forward-turn','speed40', 'speed30', 'horn')
    CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')
    
    def __init__(self, 
                 ann_file, 
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=False,
                 with_crowd=False,
                 with_label=True, 
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 split_percent=None):
        
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.img_scales = img_scale if isinstance(img_scale, 
                                                  list) else [img_scale]
        self.img_norm_cfg = img_norm_cfg
        self.with_label = with_label
        self.size_divisor = size_divisor
        self.flip_ratio = flip_ratio
        self.resize_keep_ratio = resize_keep_ratio
        self.test_mode = test_mode
        
        self.split_percent = split_percent
        
        self.img_names = []
        self.gt_bboxes = []
        self.gt_labels = []
        self.with_bboxes = []
        self.img_infos = []
        with open(self.ann_file) as f:
            lines = csv.reader(f)
            for id, line in enumerate(lines):  
                filename = line[0]
                if filename == '\ufefffilename':
                    continue
                self.img_names.append(filename)
                
                # TODO: if there is more than one label, how to handle
                self.gt_labels.append([int(line[-1])])  # [[img1], [img2],...] 
                
                coords = line[1:-1]    # 8 nums means one bbox
                if len(coords) % 8 != 0:       # error bbox
                    self.with_bboxes.append(-1)
                    self.gt_bboxes.append([0,0,0,0,0,0,0,0])
                else:
                    num_bboxes = len(coords) // 8
                    # assign bboxes num in one img
                    if num_bboxes == 0:   # empty bbox
                        self.with_bboxes.append(0)
                        self.gt_bboxes.append([0,0,0,0,0,0,0,0])
                    else:
                        self.with_bboxes.append(num_bboxes)
                        bboxes = []
                        for i in range(num_bboxes):                            
                            bboxes.append(
                                [int(coords[4*i]),int(coords[4*i+1]),
                                 int(coords[4*i+2]),int(coords[4*i+3]),
                                 int(coords[4*i+4]),int(coords[4*i+5]),
                                 int(coords[4*i+6]),int(coords[4*i+7])]) # [[x1,y1,x2,y2,x3,y3,x4,y4]]
                        self.gt_bboxes.append(bboxes)    # [[[],], [[],], [[],],...]
                # 增加一个变量用来给函数set_group_flag()作为dataloader提供
                self.img_infos.append(dict(img_id=id,
                                           filename=filename,
                                           width=3200,
                                           height=1800))
        # 去除异常数据: 但注意如果变更数据集该部分需要去掉或者修改
        self._delete_abnormal()
                
        self.gt_bboxes = np.array(self.gt_bboxes).astype(np.float32)  # (n, 1, 8)
        self.gt_labels = np.array(self.gt_labels).astype(np.int64)    # (n,1)
        self.with_bboxes = np.array(self.with_bboxes).astype(np.int64)
        
        # build transformer
        if extra_aug is not None:
            self.extra_aug = RandomFSCropTransform(**extra_aug)  # extra_aug用dict输入dict(req_size = [1333,800])
        else:
            self.extra_aug = None
        self.img_transform = ImageTransform(size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
    
    def _delete_abnormal(self):
        """通过预分析检查数据，并把异常数据去除
        error_list有标注但bbox的w/h有0或尺寸异常: 处理办法是删除对应位置的信息
        abnormal_list有标注但bbox中没有交通标志: 处理办法是对该gt_label=0 (训练集中本来没有0类，但测试集中有可能有)
        """
        error_list = [7666, 8375, 8455]  # error表示bbox不正常(w,h=0或比例不对)，删除
        abnormal_list = []               # abnormal表示数据中没有标志，bbox标注错误
        
        error_list = sorted(error_list)
        abnormal_list = sorted(abnormal_list)
        
        for i, error in enumerate(error_list):
            self.img_names.pop(error - i)      # 每删除一个，对应的位置号就会前移一位，所以要减i做补偿
            self.gt_bboxes.pop(error - i)
            self.gt_labels.pop(error - i)
        for ab in abnormal_list:
            self.gt_labels[ab] = 0
        
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
        
    def prepare_data(self, idx):
        """for formal training"""
        img = cv2.imread(os.path.join(self.img_prefix, self.img_names[idx]))
        ori_shape = (img.shape[0], img.shape[1], 3)   # h,w,(c)
        
        gt_bboxes = self.gt_bboxes[idx]   #(1,8) [[x1,y1,x2,y2,x3,y3,x4,y4]] 
        gt_bboxes = [[gt_bboxes[0][0],gt_bboxes[0][1],gt_bboxes[0][4],gt_bboxes[0][5]]]  # [[x1,y1,x3,y3]]=[[xmin,ymin,xmax,ymax]]
        gt_bboxes = np.array(gt_bboxes)   # (1,4)
        
        gt_labels = self.gt_labels[idx]     # self.gt_labels [0-20], gt_labels [0-20]
        
        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)    
        
        # basic transform
        
        img_scale = random_scale(self.img_scales)  # sample a scale from (1333,800)
        img, img_shape, pad_shape, scale_factor = self.img_transform(img,
                                                                     img_scale,
                                                                     flip=False,
                                                                     keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        gt_bboxes = self.bbox_transform(gt_bboxes, 
                                        img_shape, 
                                        scale_factor, 
                                        flip=False)
        
        # summarize the data together
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
        
        data = dict(img=DC(to_tensor(img),stack=True),
                    img_meta=DC(img_meta, cpu_only=True),
                    gt_bboxes=DC(to_tensor(gt_bboxes)),
                    gt_labels=DC(to_tensor(gt_labels)))
        return data
    
    def _prepare_data(self, idx):
        """for dataset check"""
        img = cv2.imread(os.path.join(self.img_prefix, self.img_names[idx]))
        ori_shape = (img.shape[0], img.shape[1], 3)   # h,w,(c)
        
        img_meta = dict(ori_shape=ori_shape)
        
        gt_bboxes = self.gt_bboxes[idx]
        gt_bboxes = [[gt_bboxes[0][0],gt_bboxes[0][1],gt_bboxes[0][4],gt_bboxes[0][5]]]  # [[x1,y1,x3,y3]]
        gt_bboxes = np.array(gt_bboxes)
        
        gt_labels = self.gt_labels[idx]    # self.gt_labels [1-20], gt_labels [0-19]
        
        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)    
        
        data = dict(img=img,
                    img_meta=img_meta,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels)  
        return data
        
    def __getitem__(self, idx):
        return self.prepare_data(idx)  # can be changed to _prepare_data(idx) for debug
    
    def __len__(self):
        return len(self.img_names)
    
    def summarize(self, show=False):  # 20types, 19999 imgs for training set
        total_types = []
        for i in self.gt_labels:
            total_types += i 
        total_types = len(set(total_types))
        num_error_img = len(self.with_bboxes[self.with_bboxes==-1])
        num_zero_bbox_img = len(self.with_bboxes[self.with_bboxes==0])
        num_one_bbox_img = len(self.with_bboxes[self.with_bboxes==1])
        num_two_bbox_img = len(self.with_bboxes[self.with_bboxes==2])
        num_three_bbox_img = len(self.with_bboxes[self.with_bboxes==3])
        num_four_bbox_img = len(self.with_bboxes[self.with_bboxes==4])
        
        if show:
            print("There are totally %d imgs for this dataset"%len(self.img_names))
            print("There are %d types for this dataset"%total_types)
            print("error imgs num: %d"%num_error_img)
            print("with zero bbox imgs: %d"%num_zero_bbox_img)
            print("with one bbox imgs: %d"%num_one_bbox_img)
            print("with two bboxes imgs: %d"%num_two_bbox_img)
            print("with three bboxes imgs: %d"%num_three_bbox_img)
            print("with four bboxes imgs: %d"%num_four_bbox_img)
        
        return total_types, [num_error_img, num_zero_bbox_img, 
                             num_one_bbox_img, num_two_bbox_img,
                             num_three_bbox_img, num_four_bbox_img]


class TrafficSignTest():
    """用于生成测试数据集"""
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_names = []
        for img_name in os.listdir(self.img_path):
            img_name = self.img_path + '/' + img_name
            self.img_names.append(img_name)
            
    def _prepare_data(self, idx):
        img = cv2.imread(self.img_names[idx])   # h,w,c / bgr
        ori_shape = (img.shape[0], img.shape[1])
        img_meta = dict(ori_shape=ori_shape)
        
        data = dict(img = img,
                    img_meta = img_meta)
        return data
            
    def __getitem__(self, idx):
        return self._prepare_data(idx)
        
    def __len__(self):
        return len(self.img_names)


    

    
    
    
    
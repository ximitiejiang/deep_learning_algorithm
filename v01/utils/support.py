#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 07:31:49 2019

@author: ubuntu
"""
import pickle
import torch
from dataset.utils import vis_bbox
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Debug():
    
    @staticmethod
    def save_datalist(datalist, idx, name=None):
        if name is None:
            path = './work_dirs/temp/temp_data.pkl'
        else:
            path = './work_dirs/temp/' + 'name'
        # 判断是某一类才接收
        if idx == 15:   # 15为voc的person类
            bboxes = datalist[0]
            scores = datalist[1]
            data = torch.cat([bboxes, scores[:, None]], dim=1)
            IO.save2pkl([data], path=path)
        

class IO():
    
    def __init__(self):
        pass
    
    @staticmethod
    def save2pkl(var, path, tonp=True):
        """采用pickle模块保存过程中变量，并且转换成cpu，numpy格式
        pickle模块可以保存几乎所有python对象，并将其封装保存为字符串，一般放在pkl文件中，
        可以一个pkl文件保存多个变量，从保存的pkl载入数据叫做拆封，需要按照封装的顺序拆封变量
        如果是tensor变量，不能直接存取，而需要先包装在list后再存取
        输入var可以是一个变量或一组变量列表，tonp表示将结果从tensor转化为numpy
        """
#        assert isinstance(var, list), 'the input var should be list type.'
        if tonp:
            if isinstance(var, list):
                for i in range(len(var)):
                    if isinstance(var[i], torch.Tensor):
                        var[i] = var[i].cpu().numpy() # 如果是tensor，则先转cpu和numpy
            elif isinstance(var, torch.Tensor):
                var = var.cpu().numpy()
            
        with open(path, 'wb') as f:
            pickle.dump(var, f)
    
    @staticmethod
    def save2pkl_checkidx(var_list, idx, path):
        """额外增加检查idx的功能"""
        if idx != 15:   # 15为voc的person类
            return
        
        assert isinstance(var_list, list), 'the input var should be list type.'
        for i in range(len(var_list)):
            var_list[i] = var_list[i].cpu()
            if isinstance(var_list[i], torch.Tensor):
                var_list[i] = var_list[i].numpy()
                
        with open(path, 'wb') as f:
            pickle.dump(var_list, f)
    
    @staticmethod
    def loadvar(path):
        """采用pickle模块从pkl文件读取变量清单，但需要指定有几个变量在文件中"""
        with open(path, 'rb') as f:
            var_list = pickle.load(f)
        return var_list
    
          
class DRAW():
    
    def __init__():
        pass
    
    @staticmethod
    def draw_bbox():
        pass
    
    @staticmethod
    def draw_hist(data):
        """data结构为(n,)的array
        """
        plt.figure()
        plt.title("Hist of the data")
        nums, bins, _ = plt.hist(x=data, bins=20)  # x(n,), bins为分割成多少个区间
    
    @staticmethod
    def draw_img_bbox(img_path, bboxes):
        """显示图片和外部bboxes输入"""
        if isinstance(img_path, str):
            ori_img = cv2.imread(img_path)
        if bboxes is None:
            bboxes = np.zeros((4,))
        vis_bbox(ori_img.copy(), bboxes, instance_colors=None, alpha=1., linewidth=1.5)
    
    @staticmethod
    def draw_img_bbox_inv(data, mean, std, class_names):
        """对图片和bbox进行逆变换和解包后显示
        其中data就是模型传播的数据dict，也是每个数据集的切片数据，包含('img_meta','img', 'gt_bboxes','gt_labels')
        包括：
        img: img to numpy, chw to hwc, denormalize to (0,255), rgb to bgr  
            (这里没有包含缩放操作，但缩放操作参数在img_meta里边，如果要做也可以，
            但因为bbox的缩放也没逆变换，所以img也没做，bbox的缩放逆变换在get_bbox里)
        bbox: bbox to numpy
        label: label to numpy, lable adjust position
        """
        # 图像逆变换
        img = data['img'].data.numpy()   # tensor to numpy
        img = img.transpose(1,2,0)       # chw to hwc
        img = (img * std) + mean         # denormalize to (0-255)
        img = img[...,[2,1,0]]           # rgb to bgr
        # 显示
        vis_bbox(img, data['gt_bboxes'].data.numpy(), 
                 labels=data['gt_labels'].data.numpy()-1,  # 显示时标签对照回来需要左移一位
                 class_names=class_names, 
                 instance_colors=None, alpha=1., 
                 linewidth=1.5, ax=None, saveto=None)





if __name__ == "__main__":
    aaa = 1002
    bbb = torch.tensor([21.5])
    IO.save2pkl([aaa,bbb,998], "./temp.pkl",tonp=False)
    out1 = IO.loadvar("./temp.pkl")

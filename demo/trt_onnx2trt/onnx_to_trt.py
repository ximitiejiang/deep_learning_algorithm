#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:10:13 2019

@author: ubuntu
"""
import cv2
import tensorrt as trt
from utils.onnx import inference_trt, TRTPredictor
from utils.dataset_classes import get_classes
from utils.visualization import vis_cam
"""
该示例参考tensorRT python sample
"""
class cfg():
    work_dir = '/home/ubuntu/mytrain/onnx_to_trt/'
    dataset_label_file = 'imagenet_labels.txt'
    onnx_path = '/home/ubuntu/MyWeights/onnx/resnet50v2/resnet50v2.onnx'
    imgs_path = ['/home/ubuntu/MyDatasets/misc/m1.jpg',
                 '/home/ubuntu/MyDatasets/misc/m2.jpg',
                 '/home/ubuntu/MyDatasets/misc/m3.jpg',
                 '/home/ubuntu/MyDatasets/misc/000033.jpg',
                 '/home/ubuntu/MyDatasets/misc/test11.jpg',
                 '/home/ubuntu/MyDatasets/misc/141597.jpg']  # 分别是cat,dog,bus

    gt_labels = ['cat', 'dog', 'bus','plane']
    input_size = (3, 224, 224)
    DTYPE = trt.float32
        

if __name__ == '__main__':
    task = 'cam'
                    
    if task == 'integrate':
        labels = get_classes('imagenet')
        model_path = '/home/ubuntu/mytrain/onnx_to_trt/serialized.engine'
        input_size = (224, 224) # (w, h)
        for result in inference_trt(model_path, cfg.imgs_path, input_size, labels, True):
            pred, score = result
            print('pred: %s [%.4f]'%(pred, score))    
    
    
    if task == 'cam':
        labels = get_classes('imagenet')
        model_path = '/home/ubuntu/mytrain/onnx_to_trt/serialized.engine'
        input_size = (224, 224) # (w, h)
        predictor = TRTPredictor(model_path, input_size, labels)
        vis_cam(0, predictor, labels)    
    
    
    if task == 'infe_pt': # 用pytorch推理，对比时间消耗
        from utils.evaluation import ClsPredictor   
        labels = get_classes('imagenet')
        
        cfg_path = 'cfg_classifier_resnet50_imagenet.py'
        img_id = 1
        gt_label = cfg.gt_labels[img_id]
        img = cv2.imread(cfg.imgs_path[img_id])
        predictor = ClsPredictor(cfg_path,                         
                                 load_from = None,
                                 load_device='cuda')
        pred, score = list(predictor([img]))[0]
        pred = labels[pred]
        print('pred: %s [%.4f], gt: %s'%(pred, score, gt_label))
        
        """
        比对结果：只考虑计算模型计算时间，其他图片预处理等时间都不考虑，即trt的do_inference()，和pytorch的model forward()
        1. tensorRT inference时间: 0.061s
        2. pytorch inference时间：0.0072s
        
        pytorch的计算耗时反而远小于tensorRT，可能这种普通的分类问题优化的空间很小。
        如果是计算量比较大的检测问题，可能tensorRT有一定优势。
        """
            
            
            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:42:13 2019

@author: ubuntu
"""

"""
任务：
1. 把darknet的模型和权重转换成onnx模型
2. 基于onnx模型在tensorRT进行yolov3的推断
参考：tensorRT官方： Object Detection With The ONNX TensorRT Backend In Python
"""
import tensorrt as trt

class cfg():
    work_dir = '/home/ubuntu/mytrain/onnx_yolov3/'
    
    cfg_file_path = 'https://raw.githubusercontent.com/pjreddie/darknet/f86901f6177dfc6116360a13cc06ab680e0c86b0/cfg/yolov3.cfg'
    supported_layers = ['net', 'convolutional', 'shortcut', 'route', 'upsample']
    
    
    dataset_label_file = 'imagenet_labels.txt'
    onnx_path = '/home/ubuntu/MyWeights/onnx/resnet50v2/resnet50v2.onnx'
    imgs_path = ['/home/ubuntu/MyDatasets/misc/m1.jpg',
                 '/home/ubuntu/MyDatasets/misc/m2.jpg',
                 '/home/ubuntu/MyDatasets/misc/m3.jpg',
                 '/home/ubuntu/MyDatasets/misc/000033.jpg']  # 分别是cat,dog,bus
    gt_labels = ['cat', 'dog', 'bus','plane']
    input_size = (3, 224, 224)
    DTYPE = trt.float32


class DarkNetParser():
    
    def __init__(self):
        pass
    
    def parse_cfg_file(self, cfg_file_path):
        """解析darknet的配置文件"""
        with open(cfg_file_path, 'rb') as cfg_file:  
            remainder = cfg_file.read()
            while remainder is not None:
                layer_dict, layer_name, remainder = self._next_layer(remainder)
                if layer_dict is not None:
                    self.layer_configs[layer_name] = layer_dict
        return self.layer_configs
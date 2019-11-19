#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:55:04 2019

@author: ubuntu
"""
import cv2
from utils.prepare_training import get_config, merge_config, get_model
from utils.visualization import vis_all_opencv, vis_cam
from utils.dataset_classes import get_classes
from utils.checkpoint import load_checkpoint
from utils.onnx import onnx_exporter, DetPredictorTRT

"""
用自己训练的SSD转换成onnx模型进行inference
1. ssd模型需要head头输出list/tuple/variable，不能输出dict，pytorch.onnx暂不支持
2. onnx模型导出成功，但在engine = builder.build_cuda_engine(network)报错：[TensorRT] ERROR: Network must have at least one output
应该是onnx模型导出没有output，但这个output是模型自动生成的，通过Netron可视化onnx模型，发现应该是有output
(ONNX v4, pytorch1.1, ai.onnx v9)

"""


trt_cfg = dict(model_name = 'ssd_vgg_voc.onnx',
               output_shape = [(1, 8732, 21),(1, 8732, 4)],
               postprocessor = dict(type='ssd',
                                    params = dict()),
               output_resolution = None)


def inference_onnx(src, cfg):
    predictor = DetPredictorTRT(cfg)
    
    if isinstance(src, (str, list)):  # 如果是list，则认为是图片预测
        for result in predictor(src):
            vis_all_opencv(*result, class_names=get_classes('voc'), score_thr=0.5)
    
    if isinstance(src, int):
        vis_cam(src, predictor, class_names=get_classes('voc'), score_thr=0.2)
    

if __name__ == "__main__":
    task = 'img'
        
    # cfg    
    cfg_path = './cfg_detector_ssdvgg16_voc.py'
    cfg = get_config(cfg_path)
    cfg = merge_config(trt_cfg, cfg)
    
    # 进行onnx模型导出
    if task == 'onnx':  # 
        cfg.load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth'
        onnx_exporter(cfg)
        # 检查onnx模型
        path = cfg.work_dir + cfg.model_name
        
    
    # 进行图片预测
    if task == 'img':
        # 如果是trt需要关闭tensor转换
        cfg.transform_val.img_parasms.to_tensor=False
        img_paths = ['/home/ubuntu/MyDatasets/misc/1.jpg']
        src = [cv2.imread(path) for path in img_paths]
        inference_onnx(src, cfg)
        
    # 进行摄像头预测
    if task == 'cam':
        cfg.transform_val.img_parasms.to_tensor=False
        src = 0  # cam预测    
        inference_onnx(src, cfg)
    
    
    
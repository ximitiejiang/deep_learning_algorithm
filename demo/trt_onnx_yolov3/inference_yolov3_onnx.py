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
from addict import Dict
"""
用自己训练的SSD转换成onnx模型进行inference
1. ssd模型需要head头输出list/tuple/variable，不能输出dict，pytorch.onnx暂不支持
2. onnx模型导出成功，但在engine = builder.build_cuda_engine(network)报错：[TensorRT] ERROR: Network must have at least one output
应该是onnx模型导出没有output，但这个output是模型自动生成的，通过Netron可视化onnx模型，发现应该是有output
(ONNX v4, pytorch1.1, ai.onnx v9)

"""


trt_cfg = Dict(
            model_name = 'yolov3_coco.onnx',
            load_device = 'cuda',
            img_size = (608, 608),
            work_dir = '/home/ubuntu/mytrain/onnx_yolov3/',
            transform_val = dict(
                    img_params=dict(
                            mean=[0, 0, 0],  # yolov3只做了1/255操作，不再做normalize
                            std=[1, 1, 1],
                            norm=True,
                            to_rgb=True,    # bgr to rgb
                            to_tensor=False, # 注意：trt模块只认可numpy，不能采用tensor，而是由trt自己调用tensor
                            to_chw=True,    # hwc to chw
                            flip_ratio=None,
                            scale=(608,608),  # [w,h]
                            size_divisor=None,
                            keep_ratio=False),
                    label_params=dict(
                            to_tensor=True,
                            to_onehot=None),
                    bbox_params=dict(
                            to_tensor=True
                            )),
               # 额外yolo参数
               output_shape = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)],
               postprocessor = dict(type='yolov3',
                                    params = dict(
                                        yolo_masks = [(6,7,8), (3,4,5), (0,1,2)],
                                        # YOLO anchors
                                        yolo_anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                                        (59, 119), (116, 90), (156, 198), (373, 326)],
                                        # Threshold of object confidence score (between 0 and 1)
                                        obj_threshold = 0.6,
                                        # Threshold of NMS algorithm (between 0 and 1)
                                        nms_threshold = 0.5,
                                        # Input image resolution
                                        yolo_input_resolution = (608, 608),
                                        # Number of object classes
                                        num_categories =  80)),
               output_resolution = (1000, 800))


def inference_onnx(src, cfg):
    predictor = DetPredictorTRT(cfg)
    
    if isinstance(src, (str, list)):  # 如果是list，则认为是图片预测
        for result in predictor(src):
            vis_all_opencv(*result, class_names=get_classes('coco'), score_thr=0.5)
    
    if isinstance(src, int):
        vis_cam(src, predictor, class_names=get_classes('coco'), score_thr=0.2)
    

if __name__ == "__main__":
    task = 'cam'
    
    cfg = Dict()
    cfg = merge_config(trt_cfg, cfg)
    
    if task == 'check':
        onnx_model = 
        onnx.checker.check_model(onnx_model)

    # 进行预测
    if task == 'img':
        cfg.transform_val.img_parasms.to_tensor=False
        img_paths = ['/home/ubuntu/MyDatasets/misc/3.jpg']
        src = [cv2.imread(path) for path in img_paths]
        inference_onnx(src, cfg)
    if task == 'cam':
        cfg.transform_val.img_parasms.to_tensor=False
        src = 0  # cam预测
        inference_onnx(src, cfg)
    
    
    
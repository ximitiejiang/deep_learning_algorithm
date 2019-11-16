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


trt_cfg = Dict(model_name = 'yolov3_coco',
               output_shape = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)],
               img_size = (608, 608),
               work_dir = '/home/ubuntu/mytrain/onnx_yolov3/',
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
               output_resolution = ())


def inference_onnx(src, cfg):
    predictor = DetPredictorTRT(cfg)
    
    if isinstance(src, (str, list)):  # 如果是list，则认为是图片预测
        for result in predictor(src):
            vis_all_opencv(*result, class_names=get_classes('voc'), score_thr=0.5)
    
    if isinstance(src, int):
        vis_cam(src, predictor, class_names=get_classes('voc'), score_thr=0.2)
    

if __name__ == "__main__":
    # cfg    
#    cfg_path = './cfg_detector_ssdvgg16_voc.py'
#    cfg = get_config(cfg_path)
    cfg = Dict()
    cfg.load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth'
    cfg = merge_config(trt_cfg, cfg)
    # weight to onnx
#    onnx_exporter(cfg)
    
    # 进行预测
    img_paths = ['/home/ubuntu/MyDatasets/misc/1.jpg']
#    src = 0  # cam预测
    src = [cv2.imread(path) for path in img_paths]
    
    inference_onnx(src, cfg)
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:56:40 2019

@author: ubuntu
"""
import cv2
import math
from PIL import Image
import numpy as np
from addict import Dict
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.dataset_classes import get_classes
from utils.onnx import img_loader, get_engine, allocate_buffers, do_inference
from utils.visualization import vis_all_opencv
from demo.trt_onnx_yolov3.data_processing import PreprocessYOLO, PostprocessYOLO, draw_bboxes

"""
这部分算法主要参考：Object Detection With The ONNX TensorRT Backend In Python
1. 如果直接用yolov3的原始权重转换为onnx则因为onnx从1.5版之后不再支持upsample层而导致报错。
2. 如果跳过原始权重转onnx这步，采用onnx官网上的onnx现成模型(从https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3下载)，
则卡在parser.parse(model.read())上面，报错[TensorRT] ERROR: Parameter check failed at: ../builder/Network.cpp::addInput::671, condition: isValidDims(dims, hasImplicitBatchDimension())
3. 改用trt源码的yolov3原始权重转换为onnx，又会卡在转出的onnx进行onnx.checker上，于是删除这句，然后保存onnx能够跑通。
同时采用这个版本的onnx是可以成功进行inference。所以下面的源码都是基于这个版本的onnx而不是从onnx官网下载的现成onnx.

"""

class cfg():
    work_dir = '/home/ubuntu/mytrain/onnx_yolov3/'
#    model_path = '/home/ubuntu/MyWeights/onnx/yolov3/yolov3.onnx'
    model_path = '/home/ubuntu/MyWeights/onnx/yolov3/my_trt_model/yolov3.trt'  #采用自定义的模型
    img_size = (608, 608)   # 代表输入模型的图片尺寸
#    output_shape = [1, 125, 13, 13]
    output_shape = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]   # 这里255 = 85*3, 3个anchor和每个anchor的85个预测值
    """不同yolov3版本，输出可能不同，如下都是合理的: 都是size x num_anchor x pred_items
    (3,)-> (1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)
    (3,)-> (1, 768, 85), (1, 3072, 85), (1, 12288, 85)
    (3,)-> (1, 85, 3, 19, 19), (1, 85, 3, 38, 38), (1, 85, 3, 76, 76)
    """
    
    mean=[0, 0, 0]  # yolov2在推理时没有做归一化，但yolov3做了归一化，不过只做了/255归一，没有减均值除方差，所以这里设置成mean=0，std=1
    std=[1, 1, 1]
    
    postprocess_params = dict(
            yolo_masks = [(6,7,8), (3,4,5), (0,1,2)],
            # YOLO anchors
            yolo_anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                            (59, 119), (116, 90), (156, 198), (373, 326)],
            # Threshold of object confidence score (between 0 and 1)
            obj_threshold = 0.6,
            # Threshold of NMS algorithm (between 0 and 1)
            nms_threshold = 0.5,
            # Input image resolution
            yolo_input_resolution = img_size,
            # Number of object classes
            num_categories =  80)
    

def inference_yolov3(src):
    """采用tiny yolov2的onnx模型进行预测"""
    # 对摄像头进行设置
    FPS=30
    cam_width = 1280     # 这个是摄像头的分辨率
    cam_height = 720
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    act_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 真实w,h
    act_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_info = 'Frame:%dx%d' %  (act_width, act_height)
    # 获取标签
    categories = get_classes('coco')
    # 获取logger
    logger = trt.Logger(trt.Logger.WARNING)
    # 获取engine
    engine = get_engine(cfg.model_path, logger, saveto=cfg.work_dir)
    # 创建context
    context = engine.create_execution_context()
    # 分配内存
    buffers = allocate_buffers(engine)
    # 定义预处理
#    preprocessor = PreprocessYOLO(cfg.img_size)
    postprocessor = PostprocessYOLO(**cfg.postprocess_params)
    
    task = 'cam'
    if task == 'img':
        img_path = '/home/ubuntu/MyDatasets/misc/dog.jpg'
        
        # 尝试集成加载图片的操作
        img_raw = cv2.imread(img_path)
        img_raw_size = (img_raw.shape[1], img_raw.shape[0])     # (w,h)获取原始图片尺寸, 用于与后续恢复bbox的尺寸来适配原图
        img_loader(img_raw, buffers[0][0][0], cfg.img_size, cfg.mean, cfg.std)  # resize到(608, 608), 归一化1/255，2chw, 拉直，送入hin
        results = do_inference(context, *buffers)
        
        results = [result.reshape(shape) for result, shape in zip(results, cfg.output_shape)]
        
        boxes, classes, scores = postprocessor.process(results, (img_raw_size))
        # 调整bbox的形式从(x,y,w,h)到(xmin,ymin,xmax,ymax)
        boxes[:,2:] = boxes[:, 2:] + boxes[:, :2]
        boxes = boxes.astype(np.int32)
        # 绘图
        if boxes is None:
            boxes = np.zeros((0, 4))
            labels = np.zeros((0,))
            scores = np.zeros((0,))
        
#        img = draw_bboxes(img_raw, boxes, scores, classes, categories)
        vis_all_opencv(img_raw, boxes, scores, classes, class_names=categories)
        
    
    elif task == 'cam':
        while True:
            ret, img_raw = cap.read()
            if ret != True:
                continue
            # 检查是否有按键中断
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
            # img loader之后图片会送入GPU
            _ = img_loader(img_raw, buffers[0][0][0], cfg.img_size, cfg.mean, cfg.std)  # chw
            # 开始预测
            results = do_inference(context, *buffers)
            # 对结果进行处理
            results = [result.reshape(shape) for result, shape in zip(results, cfg.output_shape)]  # 把得到的GPU展平数据恢复形状[(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)], 255 = 25*5
            bboxes, labels, scores = postprocessor.process(results, (cam_width, cam_height))  # (k,4), (k,), (k,), 图片会被放大到cam_width, cam_height
            # 调整bbox的形式从(x,y,w,h)到(xmin,ymin,xmax,ymax)
            bboxes[:,2:] = bboxes[:, 2:] + bboxes[:, :2]
            bboxes = bboxes.astype(np.int32)
            # 绘图
            if bboxes is None:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0,))
                scores = np.zeros((0,))
            vis_all_opencv(img_raw, bboxes, scores, labels,
                           win_name=frame_info,
                           class_names=categories, 
                           score_thr=0.1)
        cap.release()

if __name__ == "__main__":
    inference_yolov3(src=0)
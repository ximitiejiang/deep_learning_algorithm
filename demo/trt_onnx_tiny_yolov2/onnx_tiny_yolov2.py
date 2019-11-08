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
from demo.trt_onnx_tiny_yolov2.post_process import PostprocessYOLO

"""
这部分算法主要参考：https://github.com/tsutof/tiny_yolov2_onnx_cam
1. 手动下载yolov2的onnx模型：https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov2
2. 指定模型对应的标签：该onnx模型是在voc数据集下训练得到，所以要获得voc的标签列表
3. 指定该模型的输入图片尺寸(416,416)，以及输出尺寸(1,125,13,13)
4. 指定该模型对输入图片的预处理过程：注意inference不需要对img进行归一化。
    - resize到(416,416)
5. 
"""



class cfg():
    work_dir = '/home/ubuntu/mytrain/onnx_tiny_yolov2/'
#    model_path = '/home/ubuntu/MyWeights/onnx/tiny_yolov2/Model.onnx'
    model_path = work_dir + 'serialized.engine'
    img_size = (416, 416)   # 代表输入模型的图片尺寸
    output_shape = [1, 125, 13, 13]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    postprocess_params = dict(
            yolo_masks = [(0, 1, 2, 3, 4)],
            # YOLO anchors
            yolo_anchors = [(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)],
            # Threshold of object confidence score (between 0 and 1)
            obj_threshold = 0.001,
            # Threshold of NMS algorithm (between 0 and 1)
            nms_threshold = 0.3,
            # Input image resolution
            yolo_input_resolution = img_size,
            # Number of object classes
            num_categories =  20)
    

def inference_img():
    pass

def inference_cam_tiny_yolov2(src):
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
    categories = get_classes('voc')
    # 获取logger
    logger = trt.Logger(trt.Logger.WARNING)
    # 获取engine
    engine = get_engine(cfg.model_path, logger, saveto=cfg.work_dir)
    # 创建context
    context = engine.create_execution_context()
    # 分配内存
    buffers = allocate_buffers(engine)
    # 定义预处理
    postprocessor = PostprocessYOLO(**cfg.postprocess_params)
    while True:
        ret, img_raw = cap.read()
        if ret != True:
            continue
        # 检查是否有按键中断
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
        # img loader之后图片会送入GPU
        _ = img_loader(img_raw, buffers.hin, cfg.img_size, None, None)  # 注意yolov2 inference不需要normalize
        do_inference(buffers, context)
        # 对结果进行处理
        hout = [buffers.hout.reshape(cfg.output_shape)]  # 把得到的GPU展平数据恢复形状(1,125,13,13), 同时放入list中作为多个特征图的一张，只不过这里只使用了一张特征图
        bboxes, labels, scores = postprocessor.process(hout, (cam_width, cam_height))  # (k,4), (k,), (k,), 图片会被放大到cam_width, cam_height
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
                       score_thr=0.3)
    cap.release()

if __name__ == "__main__":
    inference_cam_tiny_yolov2(src=0)
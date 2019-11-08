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
from demo.trt_onnx_yolov3.data_processing import PreprocessYOLO, PostprocessYOLO

"""
这部分算法主要参考：Object Detection With The ONNX TensorRT Backend In Python
0. 忽略了把yolov3原本的权重转化为onnx的过程，因为onnx高版本已经取消支持upsample，只能用onnx<1.4的版本来转换，这里偷懒直接用onnx官方repo里边现成onnx.
1. 手动下载yolov3的onnx模型：https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3
2. 卡在parser.parse(model.read())上面，报错[TensorRT] ERROR: Parameter check failed at: ../builder/Network.cpp::addInput::671, condition: isValidDims(dims, hasImplicitBatchDimension())
怀疑还是因为model里边包含upsample而我的onnx不支持? 但这是tensorRT报错阿，难道tensorRT也不支持?
"""

class cfg():
    work_dir = '/home/ubuntu/mytrain/onnx_yolov3/'
    model_path = '/home/ubuntu/MyWeights/onnx/yolov3/yolov3.onnx'
#    model_path = work_dir + 'serialized.engine'
    img_size = (608, 608)   # 代表输入模型的图片尺寸
#    output_shape = [1, 125, 13, 13]
    output_shape = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
    
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
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
    preprocessor = PreprocessYOLO(cfg.img_size)
    postprocessor = PostprocessYOLO(**cfg.postprocess_params)
    
    task = 'cam'
    if task == 'img':
        """图片预处理方式：先resize到hwc(608,608,3), 然后除以255， 然后变换到chw,然后变换到(1,c,h,w),"""
        img_path = '/home/ubuntu/MyDatasets/misc/dog.jpg'
        img_raw, img = preprocessor(img_path)
        
    
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
            _ = img_loader(img_raw, buffers.hin, cfg.img_size, cfg.mean, cfg.std)  # chw
            # 开始预测
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
                           score_thr=0.1)
        cap.release()

if __name__ == "__main__":
    inference_yolov3(src=0)
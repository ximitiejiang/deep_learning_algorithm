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
from utils.onnx import img_loader
from utils.visualization import vis_all_opencv
from demo.trt_onnx_tiny_yolov2.post_process import PostprocessYOLO

class cfg():
    model_path = '/home/ubuntu/MyWeights/onnx/tiny_yolov2/Model.onnx'
    img_size = (416, 416)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    post_params = dict(
            yolo_masks = [(0, 1, 2, 3, 4)],
            # YOLO anchors
            yolo_anchors = [(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)],
            # Threshold of object confidence score (between 0 and 1)
            obj_threshold = 0.6,
            # Threshold of NMS algorithm (between 0 and 1)
            nms_threshold = 0.3,
            # Input image resolution
            yolo_input_resolution = img_size,
            # Number of object classes
            num_categories =  20)
    

def inference_tiny_yolov2(src):
    """采用tiny yolov2的onnx模型进行预测"""
    # 对摄像头进行设置
    FPS=30
    width = 1280
    height = 720
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    act_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 真实w,h
    act_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_info = 'Frame:%dx%d' %  (act_width, act_height)
    # 获取标签
    labels = get_classes('voc')
    # 获取logger
    logger = trt.Logger(trt.Logger.WARNING)
    # 获取engine
    if cfg.model_path.split('.')[-1] == 'engine':
        with open(cfg.model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    elif cfg.model_path.split('.')[-1] == 'onnx':
        with trt.Builder(logger) as builder, builder.create_network() as network, trt.OnnxParser(network, logger) as parser: # with + 局部变量便于释放内存
            builder.max_workspace_size = 1*1 << 20
            builder.max_batch_size = 1
            with open(cfg.model_path, 'rb') as model:  # 打开onnx
                parser.parse(model.read())       # 读取onnx, 解析onnx(解析的过程就是把权重填充到network的过程)
                engine = builder.build_cuda_engine(network)  # 这个过程包括构建network层，填充权重，优化计算过程需要一定耗时
    # 创建context
    context = engine.create_execution_context()
    # 分配内存
    hin = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))  # (c*h*w,)把输入图片拉直的一维数组
    hout = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32)) # (n_cls,)输出预测的一维数组
    din = cuda.mem_alloc(hin.nbytes)    #　为GPU设备分配内存
    dout = cuda.mem_alloc(hout.nbytes)    
    buffers = Dict(hin=hin, hout=hout, din=din, dout=dout)
    stream = cuda.Stream()              # 创建stream流来拷贝输入输出，进行推理  
    #开始预测
    postprocessor = PostprocessYOLO(**cfg.post_params)
    while True:
        ret, img_raw = cap.read()
        if ret != True:
            continue
        # 检查是否有按键中断
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
#        img = cv2.resize(img, cfg.img_size)   # 图片尺寸resize到(416,416)
#        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # bgr to rgb
        img = img_loader(img_raw, buffers.hin, cfg.img_size, cfg.mean, cfg.std)  # chw
        img = img[None,...]  # (1,3,416,416)
        # do inference
        cuda.memcpy_htod_async(buffers.din, buffers.hin, stream)  # 数据从host(cpu)送入device(GPU)
        context.execute_async(bindings=[int(buffers.din), int(buffers.dout)], stream_handle=stream.handle)  # 执行推断
        cuda.memcpy_dtoh_async(buffers.hout, buffers.dout, stream)# 把预测结果从GPU返回cpu: device to host
        stream.synchronize()
        # 对结果进行处理
        bboxes, labels, scores = postprocessor.process(buffers.hout, (width, height))  # (k,4), (k,), (k,)
        # 绘图
        vis_all_opencv(img_raw, bboxes, labels, scores,
                       win_name=frame_info,
                       class_names=labels, 
                       score_thr=0.2)
        

if __name__ == "__main__":
    inference_tiny_yolov2(src=0)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:45:33 2019

@author: ubuntu
"""
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.tools import timer 
from utils.transform import imresize


def img_loader(img_raw, pagelocked_buffer, input_size):
    """加载单张图片，进行归一化，并放入到锁页内存"""
#    img_raw = Image.open(img_path)    # rgb (Image读入的直接就是rgb格式了，所以无需再从bgr2rgb)
#    img_raw = cv2.imread(img_path)
    w, h = input_size
    img_raw = imresize(img_raw, (w, h))  # hwc, bgr
    
#    img_raw = np.asarray(img_raw.resize((w,h), Image.ANTIALIAS))  #(h,w,c) rgb
    img = img_raw[...,[2, 1, 0]].transpose([2, 0, 1]).astype(trt.nptype(trt.float32)) # chw, rgb
    img = (img/255.0 - 0.45) / 0.225   # 归一化：类似pytorch的模式，先归一到[0,1]然后norm到标准正态分布, 采用imagenet的标准mean=(0.45,0.45,0.45),std=(0.225,0.225,0.225)
    np.copyto(pagelocked_buffer, img.reshape(-1,))  # 展平送入GPU
    return img_raw


def get_engine(model_path, logger, saveto=None):
    """把模型转化为trt engine"""
    
    # 如果是序列化模型，则直接逆序列化即可
    if model_path.split('.')[-1] == 'engine':
        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    
    # 如果是onnx模型，则需要先转化为engine    
    elif model_path.split('.')[-1] == 'onnx':
        with trt.Builder(logger) as builder, builder.create_network() as network, trt.OnnxParser(network, logger) as parser: # with + 局部变量便于释放内存
            builder.max_workspace_size = 1*1 << 20
            builder.max_batch_size = 1
            with open(model_path, 'rb') as model:  # 打开onnx
                parser.parse(model.read())       # 读取onnx, 解析onnx(解析的过程就是把权重填充到network的过程)
                engine = builder.build_cuda_engine(network)  # 这个过程包括构建network层，填充权重，优化计算过程需要一定耗时
                if saveto is not None:
                    with open(saveto + 'serialized.engine', 'wb') as f:
                        f.write(engine.serialize())
    else:
        raise ValueError('not supported model type.')
    return engine


def softmax(x):
    """numpy版本softmax函数, x(m,)为一维数组"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 在np.exp(x - C), 相当于对x归一化，防止因x过大导致exp(x)无穷大使softmax输出无穷大 
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def inference_trt(model_path, img_path, input_size, labels=None, show=None):
    """汇总一个输入onnx就能进行分类预测的子程序，针对单张图片
    args:
        model_path: str, 可以是onnx模型(转化为engine)，也可以是序列化模型(则直接逆序列化加载)
        img_path: (n, ) list of paths
    return:
        pred: int, 从n~n_cls-1的值
        score: float, 
    """
    logger = trt.Logger(trt.Logger.WARNING)
    with get_engine(model_path, logger) as engine:  # 创建engine
        # 分配内存
        hin = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))  # (c*h*w,)把输入图片拉直的一维数组
        hout = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32)) # (n_cls,)输出预测的一维数组
        din = cuda.mem_alloc(hin.nbytes)    #　为GPU设备分配内存
        dout = cuda.mem_alloc(hout.nbytes)    
        stream = cuda.Stream()              # 创建stream流来拷贝输入输出，进行推理
        # 创建context
        with engine.create_execution_context() as context:
            if isinstance(img_path, str):
                img_path = [img_path]
            for path in img_path:
                img_raw = img_loader(path, hin, input_size)
                # do inference
                cuda.memcpy_htod_async(din, hin, stream)
                with timer("trt inference time"):
                    context.execute_async(bindings=[int(din), int(dout)], stream_handle=stream.handle)  # 执行推断
                    cuda.memcpy_dtoh_async(hout, dout, stream)# 把预测结果从GPU返回cpu: device to host
                    stream.synchronize()
                pred = np.argmax(hout)   # 第几个label
                score = np.max(softmax(hout))
                if labels is not None:
                    pred = labels[pred]
                if show:
                    text1 = 'PREDS: ' + str(pred)
                    text2 = 'SCORE: ' + str(score)
                    cv2.putText(img_raw, text1, (5, 10),
                                cv2.FONT_HERSHEY_DUPLEX, 
                                0.4, 
                                [255,0,0])
                    cv2.putText(img_raw, text2, (5, 20),
                                cv2.FONT_HERSHEY_DUPLEX, 
                                0.4, 
                                [255,0,0]) # bgr, 所以蓝色要用255,0,0，而不是0,0,255
                    cv2.imshow(text1, img_raw)
    yield pred, score


class TRTPredictor():
    """为了跟之前的摄像头等兼容，wrap到predictor类里边取"""    
    def __init__(self, model_path, input_size, labels=None):
        self.type = 'trt'
        self.model_path = model_path
        self.input_size = input_size
        self.labels = labels
        self.logger = trt.Logger(trt.Logger.WARNING)
        
    def __call__(self, src):
        with get_engine(self.model_path, self.logger) as engine:  # 创建engine
            # 分配内存
            hin = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))  # (c*h*w,)把输入图片拉直的一维数组
            hout = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32)) # (n_cls,)输出预测的一维数组
            din = cuda.mem_alloc(hin.nbytes)    #　为GPU设备分配内存
            dout = cuda.mem_alloc(hout.nbytes)    
            stream = cuda.Stream()              # 创建stream流来拷贝输入输出，进行推理
            # 创建context
            with engine.create_execution_context() as context:
                if isinstance(src, np.ndarray):
                    src = [src]
                for img in src:
                    img_raw = img_loader(img, hin, self.input_size)
                    # do inference
                    cuda.memcpy_htod_async(din, hin, stream)
                    with timer("trt inference time"):
                        context.execute_async(bindings=[int(din), int(dout)], stream_handle=stream.handle)  # 执行推断
                        cuda.memcpy_dtoh_async(hout, dout, stream)# 把预测结果从GPU返回cpu: device to host
                        stream.synchronize()
                    pred = np.argmax(hout)   # 第几个label
                    score = np.max(softmax(hout))
                    if self.labels is not None:
                        pred = self.labels[pred]
                    text1 = 'PREDS: ' + str(pred)
                    text2 = 'SCORE: ' + str(score)
                    cv2.putText(img_raw, text1, (5, 10),
                                cv2.FONT_HERSHEY_DUPLEX, 
                                0.4, 
                                [255,0,0])
                    cv2.putText(img_raw, text2, (5, 20),
                                cv2.FONT_HERSHEY_DUPLEX, 
                                0.4, 
                                [255,0,0]) # bgr, 所以蓝色要用255,0,0，而不是0,0,255
                    yield img_raw, pred, score        
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
from addict import Dict
from utils.tools import timer 
from utils.transform import imresize, imnormalize


def img_loader(img_raw, pagelocked_buffer, input_size, 
               mean=None, std=None):
    """加载单张图片，进行归一化，并放入到锁页内存
    注意：这里的锁页内存就是用来存放输入图片的，所以是allocate buffer里边的hin
    """
#    img_raw = Image.open(img_path)    # rgb (Image读入的直接就是rgb格式了，所以无需再从bgr2rgb)
#    img_raw = cv2.imread(img_path)
    w, h = input_size
    img_raw = imresize(img_raw, (w, h))  # hwc, bgr
    
    img = img_raw[...,[2, 1, 0]]; # 先to rgb
    if mean is not None:
        img = imnormalize(img / 255, mean, std)  # 再归一化 hwc,rgb
    img = img.transpose([2, 0, 1]).astype(trt.nptype(trt.float32)) # chw, rgb

    np.copyto(pagelocked_buffer, img.reshape(-1,))  # (c*h*w, )展平送入GPU，这一步所有的都要做，有的做法是直接hin = img(即因为分配的内存hin是展平的，所把不展平的img放入也会自动展平)
    return img_raw


def get_engine(model_path, logger=None, saveto=None):
    """把模型转化为trt engine
    1. 如果是onnx模型，则采用trt自带的OnnxParser先解析onnx模型(把权重放入network中)，然后采用trt自带builder把network转换成engine.
    理论上说，只要模型的层是trt支持的，任何onnx模型都是可以自动转换成engine的。
    2. 如果已经是序列化engine，则直接反序列化后加载。
    """
    if logger is None:
        logger = trt.Logger(trt.Logger.WARNING)
    # 如果是序列化模型，则直接逆序列化即可(序列化模型建议的后缀名是.engine或.trt)
    if model_path.split('.')[-1] in ['engine', 'trt']:
        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    
    # 如果是onnx模型，则需要先转化为engine    
    elif model_path.split('.')[-1] == 'onnx':
        print('start transfer onnx model...')
        with trt.Builder(logger) as builder, builder.create_network() as network, trt.OnnxParser(network, logger) as parser: # with + 局部变量便于释放内存
            builder.max_workspace_size = 1*1 << 30  # 注意：2^10代表1024也就是1k, 所以2^20代表1M, 2^30代表1G (一般GPU的内存够大的建议都定义成1G) 
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


def allocate_buffers(engine):
    # 分配内存： 分配内存可以实现分配，因为他只分配一次，是所有图片共享
    hin = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))  # (c*h*w,)把输入图片拉直的一维数组
    hout = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32)) # (n_cls,)输出预测的一维数组
    din = cuda.mem_alloc(hin.nbytes)    #　为GPU设备分配内存
    dout = cuda.mem_alloc(hout.nbytes)    
    stream = cuda.Stream()              # 创建stream流来拷贝输入输出，进行推理
    buffers = Dict(hin=hin, hout=hout, din=din, dout=dout, stream=stream)
    return buffers


def do_inference(buffers, context):
    # 进行推断，最后的结果就在buffers.hout
    cuda.memcpy_htod_async(buffers.din, buffers.hin, buffers.stream)  # 数据从host(cpu)送入device(GPU)
    context.execute_async(bindings=[int(buffers.din), int(buffers.dout)], stream_handle=buffers.stream.handle)  # 执行推断
    cuda.memcpy_dtoh_async(buffers.hout, buffers.dout, buffers.stream)# 把预测结果从GPU返回cpu: device to host
    buffers.stream.synchronize()  # 同步


def softmax(x):
    """numpy版本softmax函数, x(m,)为一维数组"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 在np.exp(x - C), 相当于对x归一化，防止因x过大导致exp(x)无穷大使softmax输出无穷大 
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class DetTRTPredictor():
    def __init__(self, model_path, input_size, labels, output_shape, postprocessor, resolution):
        self.type = 'det'
        self.model_path = model_path
        self.input_size = input_size
        self.labels = labels
        self.output_shape = output_shape    # 代表输出特征图尺寸
        self.logger = trt.Logger(trt.Logger.WARNING)
        # 创建engine
        self.engine = get_engine(self.model_path, self.logger)
        # 创建context
        self.context = self.engine.create_execution_context()
        # 分配内存： 分配内存可以实现分配，因为他只分配一次，是所有图片共享
        self.buffers = allocate_buffers(self.engine)
        self.postprocessor = postprocessor
        self.resolution = resolution
    
    def __call__(self, src):
        # 开始预测
        if isinstance(src, np.ndarray):
            src = [src]
        for img in src:
            img_raw = img_loader(img, self.buffers.hin, self.input_size)
            # do inference
            do_inference(self.buffers, self.context)
            # 结果解析
            hout = [self.buffers.hout.reshape(self.output_shape)]  # 把得到的GPU展平数据恢复形状(1,125,13,13), 同时放入list中作为多个特征图的一张，只不过这里只使用了一张特征图
            bboxes, labels, scores = self.postprocessor.process(hout, self.resolution)  # (k,4), (k,), (k,), 图片会被放大到cam_width, cam_height
            # 调整bbox的形式从(x,y,w,h)到(xmin,ymin,xmax,ymax)
            bboxes[:,2:] = bboxes[:, 2:] + bboxes[:, :2]
            bboxes = bboxes.astype(np.int32)
            
            if bboxes is None:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0,))
                scores = np.zeros((0,))

            yield img_raw, bboxes, score           
        
    

class ClsTRTPredictor():
    """采用tensorRT进行分类模型的预测：为了跟之前的摄像头等兼容，wrap到predictor类里边取
    args:
        model_path: str, 可以是onnx模型或者是
    """    
    def __init__(self, model_path, input_size, labels=None):
        self.type = 'cls'
        self.model_path = model_path
        self.input_size = input_size
        self.labels = labels
        self.logger = trt.Logger(trt.Logger.WARNING)
        # 创建engine
        self.engine = get_engine(self.model_path, self.logger)
        # 创建context
        self.context = self.engine.create_execution_context()
        # 分配内存： 分配内存可以实现分配，因为他只分配一次，是所有图片共享
        self.buffers = allocate_buffers(self.engine)
        
    def __call__(self, src):
        # 开始预测
        if isinstance(src, np.ndarray):
            src = [src]
        for img in src:
            img_raw = img_loader(img, self.buffers.hin, self.input_size)
            # do inference
            do_inference(self.buffers, self.context)
            # 结果解析
            pred = np.argmax(self.buffers.hout)   # 第几个label
            score = np.max(softmax(self.buffers.hout))
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
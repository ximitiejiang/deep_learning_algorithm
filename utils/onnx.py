#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:45:33 2019

@author: ubuntu
"""
import numpy as np
import torch
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from addict import Dict
from utils.tools import timer 
from utils.transform import imresize, imnormalize
from utils.prepare_training import get_config, get_model
from utils.checkpoint import load_checkpoint

def onnx_exporter(cfg, output_path):
    """把一个pytorch模型转换成onnx模型，对模型的要求：
    1. 模型需要有forward_dummy()函数的实施，如下是一个实例：
    def forward_dummy(self, img):
        x = self.extract_feat(img)
        x = self.bbox_head(x)
        return x
    2. 模型的终端输出即head端的输出必须是tuple/list/variable类型，不能是dict，否则当前pytorch.onnx不支持导出。
    """
    img_shape = (1, 3) + cfg.img_size
    dummy_input = torch.randn(img_shape, device='cuda')
    # 创建配置和创建模型
    model = get_model(cfg).cuda()
    if cfg.load_from is not None:
        _ = load_checkpoint(model, cfg.load_from)
    else:
        raise ValueError('need to assign checkpoint path to load from.')
    # 指定模型的前向通道
    model.forward = model.forward_dummy
    # 导出模型
    torch.onnx.export(model, dummy_input, output_path, verbose=True)
    

def img_loader(img_raw, pagelocked_buffer, input_size, 
               mean=None, std=None):
    """加载单张图片，进行归一化，并放入到锁页内存
    注意：这里的锁页内存就是用来存放输入图片的，所以是allocate buffer里边的hin
    同时锁页内存中的数据只能送入拉直展平的一维数组。
    """
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
    """预分配GPU的计算内存：engine.get_binding_shape获得的是engine的输入输出形状
    returns
        inputs: (1,) (hin, din), 一般代表一路展平的输入，比如hin=[(1108992,)], din一般是nbytes看不到，但可通过in(din)看到一个数值
        outputs: (n,) (hout, dout), 一般代表多路展平的输出，比如hout = [(92055,)(368220,)(1472880,)]
        bindings: (1+n,) (int(din or dout))
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append([host_mem, device_mem])
        else:
            outputs.append([host_mem, device_mem])
    return [inputs, outputs, bindings, stream]      
    
#    dins = []
#    hins = []
#    douts = []
#    houts = []
#    for binding in engine:
#        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size)
#        dtype = trt.nptype(trt.float32)
#        hin = cuda.pagelocked_empty(size, dtype
#        hins.append(hin)   # (c*h*w,)把输入图片拉直的一维数组
#        houts.append(cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))) # (n_cls,)输出预测的一维数组
#        dins.append(cuda.mem_alloc(hin.nbytes))    #　为GPU设备分配内存
#        douts.append(cuda.mem_alloc(hout.nbytes))    
#    stream = cuda.Stream()              # 创建stream流来拷贝输入输出，进行推理
#    buffers = Dict(hin=hins, hout=houts, din=dins, dout=douts, stream=stream)  # 字典中每个元素都是list
#    return buffers


def do_inference(context, inputs, outputs, bindings, stream, batch_size=1):
    """如果要用trt做推断，则模型返回的每层特征图的预测结构都必须是单个数据，比如3层特征图，就输出3个tensor，每个tensor包含了pred(类别), score(置信度),bbox(坐标) 
    这样便于trt内部进行展平计算，因为展平计算有利于卷积计算优化(是整个计算的核心)
    """
    # 进行推断，最后的结果就在buffers.hout
    [cuda.memcpy_htod_async(input[1], input[0], stream) for input in inputs]   # 数据从host(cpu)送入device(GPU)
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle) # 执行推断
    [cuda.memcpy_dtoh_async(output[0], output[1], stream) for output in outputs]# 把预测结果从GPU返回cpu: device to host
    stream.synchronize()  # 同步
    return [output[0] for output in outputs]   # (n,)返回输出的hout，即展平的计算结果

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
            img_raw = img_loader(img, self.buffers[0][0], self.input_size)  #加载到hin
            # do inference
            trt_result = do_inference(self.context, *self.buffers)
            # 结果解析
            hout = [trt_result.reshape(self.output_shape)]  # 把得到的GPU展平数据恢复形状(1,125,13,13), 同时放入list中作为多个特征图的一张，只不过这里只使用了一张特征图
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
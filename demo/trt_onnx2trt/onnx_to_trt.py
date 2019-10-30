#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:10:13 2019

@author: ubuntu
"""
import numpy as np
import random
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.tools import timer 

"""
该示例参考tensorRT python sample
"""
class cfg():
    work_dir = '/home/ubuntu/mytrain/onnx_to_trt/'
    dataset_label_file = 'imagenet_labels.txt'
    onnx_path = '/home/ubuntu/MyWeights/onnx/resnet50v2/resnet50v2.onnx'
    imgs_path = ['/home/ubuntu/MyDatasets/misc/m1.jpg',
                 '/home/ubuntu/MyDatasets/misc/m2.jpg',
                 '/home/ubuntu/MyDatasets/misc/m3.jpg',
                 '/home/ubuntu/MyDatasets/misc/000033.jpg']  # 分别是cat,dog,bus
    gt_labels = ['cat', 'dog', 'bus','plane']
    input_size = (3, 224, 224)
    DTYPE = trt.float32
        

def img_loader(img_path, pagelocked_buffer):
    """加载单张图片，进行归一化，并放入到锁页内存"""
    img = Image.open(img_path)    # 
    c, h, w = cfg.input_size
    img = np.asarray(img.resize((w,h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(cfg.DTYPE)) # (c,h,w)numpy
    img = (img/255.0 - 0.45) / 0.225   # 归一化：类似pytorch的模式，先归一到[0,1]然后norm到标准正态分布, 采用imagenet的标准mean=(0.45,0.45,0.45),std=(0.225,0.225,0.225)
    np.copyto(pagelocked_buffer, img.reshape(-1,))
        

def get_engine(onnx_path, logger, saveto=None):
    """把onnx模型转化为trt engine"""
    with trt.Builder(logger) as builder, builder.create_network() as network, trt.OnnxParser(network, logger) as parser: # with + 局部变量便于释放内存
        builder.max_workspace_size = 1*1 << 20
        builder.max_batch_size = 1
        with open(onnx_path, 'rb') as model:  # 打开onnx
            parser.parse(model.read())       # 读取onnx, 解析onnx(解析的过程就是把权重填充到network的过程)
            engine = builder.build_cuda_engine(network)  # 这个过程包括构建network层，填充权重，优化计算过程需要一定耗时
            if saveto is not None:
                with open(saveto + 'serialized.engine', 'wb') as f:
                    f.write(engine.serialize())
            else:
                return engine
                    

def allocate_buffers(engine):
    """为engine进行优化过程中产生的激活值分配内存: 
    其中binding_shape(0)=(3,224,224),binding_shape(1)=(1000,)，表示输入形式(c,h,w)以及最终输出的形式(1000类的score)
    """
#    size = trt.volume(engine.get_binding_shape(0)) * engine.max_batch_size
#    dtype = trt.nptype(engine.get_binding_dtype(0))
    # 为主机分配锁页内存
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(cfg.DTYPE))  # (c*h*w,)把输入图片拉直的一维数组
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(cfg.DTYPE)) # (n_cls,)输出预测的一维数组
    #　为设备分配内存
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)    
    # 创建stream流来拷贝输入输出，进行推理
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream



def do_inference(context, h_input, d_input, h_output, d_output, stream):
    """对单个样本(batch_size=1)进行推断"""
    cuda.memcpy_htod_async(d_input, h_input, stream)  #把输入送入GPU： host to device
    with timer('trt_inference'):
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)  # 执行推断
        cuda.memcpy_dtoh_async(h_output, d_output, stream)# 把预测结果从GPU返回cpu: device to host
        stream.synchronize()   # 流同步


def imagenet_labels(label_file):
    """获得image label"""
    labels = []
    with open(label_file) as f:
        lines = f.readlines()   # 1000行标签
        for line in lines:
            label = line[9:-1]
#            label = line[9:-1].split(',')
#            label = [l[1:] for l in label]
            labels.append(label)
    return labels


def softmax(x):
    """softmax函数, x(m,)为一维数组"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 在np.exp(x - C), 相当于对x归一化，防止因x过大导致exp(x)无穷大使softmax输出无穷大 
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    

if __name__ == '__main__':
    task = 'infe'
    
    if task == 'debug':
        logger = trt.Logger(trt.Logger.WARNING)                 # 构造logger
        labels = imagenet_labels(cfg.dataset_label_file)
        with get_engine(cfg.onnx_path, logger) as engine:
            buffers = allocate_buffers(engine)
            with engine.create_execution_context() as context:  # 构造context
                img_id = random.choice(range(len(cfg.imgs_path)))
                img_id = 0
                gt_label = cfg.gt_labels[img_id]
                img_loader(cfg.imgs_path[img_id], buffers[0])   # 获取图片,并放入锁页内存
                do_inference(context, *buffers)                 # 进行推理
                pred = labels[np.argmax(buffers[2])]
                score = np.max(softmax(buffers[2]))
                print('pred: %s [%.4f], gt: %s'%(pred, score, gt_label))
                    
    
    if task == 'serialize':  # get engine
        logger = trt.Logger(trt.Logger.WARNING)                 # 构造logger
        get_engine(cfg.onnx_path, logger, saveto=cfg.work_dir)
    
    
    if task == 'infe_trt':
        logger = trt.Logger(trt.Logger.WARNING)                 # 构造logger
        labels = imagenet_labels(cfg.dataset_label_file)

        with open(cfg.work_dir + 'serialized.engine', 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            buffers = allocate_buffers(engine)                   # 分配内存： [hin, hout, din, dout, stream]
            with engine.create_execution_context() as context:  # 构造context
                img_id = random.choice(range(len(cfg.imgs_path)))
                img_id = 2
                gt_label = cfg.gt_labels[img_id]
                img_loader(cfg.imgs_path[img_id], buffers[0])   # 获取图片,并放入锁页内存
                do_inference(context, *buffers)                 # 进行推理
                pred = labels[np.argmax(buffers[2])]
                score = np.max(softmax(buffers[2]))
                print('pred: %s [%.4f], gt: %s'%(pred, score, gt_label))
        
    if task == 'infe_pt': # 用pytorch推理，对比时间消耗
        from utils.evaluation import ClsPredictor
        import cv2
        
        labels = imagenet_labels(cfg.dataset_label_file)
        
        cfg_path = 'cfg_classifier_resnet50_imagenet.py'
        img_id = 1
        gt_label = cfg.gt_labels[img_id]
        img = cv2.imread(cfg.imgs_path[img_id])
        predictor = ClsPredictor(cfg_path,                         
                                 load_from = None,
                                 load_device='cuda')
        pred, score = list(predictor([img]))[0]
        pred = labels[pred]
        print('pred: %s [%.4f], gt: %s'%(pred, score, gt_label))
        
        """
        比对结果：只考虑计算模型计算时间，其他图片预处理等时间都不考虑，即trt的do_inference()，和pytorch的model forward()
        1. tensorRT inference时间: 0.061s
        2. pytorch inference时间：0.0072s
        
        pytorch的计算耗时反而远小于tensorRT，可能这种普通的分类问题优化的空间很小。
        如果是计算量比较大的检测问题，可能tensorRT有一定优势。
        """
            
            
            
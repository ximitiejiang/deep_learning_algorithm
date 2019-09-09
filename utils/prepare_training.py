#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:50:43 2019

@author: ubuntu
"""
import sys, os
from addict import Dict
from importlib import import_module
import logging
import torch
import numpy as np

from utils.tools import get_time_str

# %%
def get_config(config_path="cfg_ssd_voc.py"):
    """从py文件中获取配置信息，py文件中需要为变量赋值形式数据，导入为Dict类型"""
    file_name = os.path.abspath(os.path.expanduser(config_path))  # 获得完整路径
    module_name = os.path.basename(file_name)[:-3] # 取名字
    dir_name = os.path.dirname(file_name)
    # 导入module时需要确保路径在sys.path下才能成功导入
    if(not dir_name in sys.path):
        sys.path.insert(0, dir_name)
    cfg_data = import_module(module_name)   # 导入后python中的module对象：python中的包就是文件夹，module就是文件，相当于导入整个py文件        
    cfg_dict = {}
    for name, value in cfg_data.__dict__.items():  # 从module对象中提取内部__dict__(python中一切皆对象，__dict__用来存放对象的属性，所以只要.py文件中都写成变量形式，就能在__dict__中找到)
        if not name.startswith("__"):
            cfg_dict[name] = value
    return Dict(cfg_dict) # 返回Dict,从而可以类似属性调用


# %%
def get_logger(logger_cfg):
    """创建logger"""
    # 先定义format/level
    format = "%(asctime)s - %(levelname)s - %(message)s"  # 用于logger的输出前缀
    logging.basicConfig(format=format)
    # 再创建logger
    logger = logging.getLogger()
    logger.setLevel(logger_cfg.log_level) # 必须手动设置一次level,否则logger是默认类型(可通过logger.level查看)
    log_dir = logger_cfg.get('log_dir', None)
    if log_dir is not None:
        filename = '{}.log'.format(get_time_str())
        log_file = os.path.join(log_dir, filename)
        # 创建文件句柄
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logger_cfg.log_level)
        # 添加文件句柄
        logger.addHandler(file_handler)
    return logger


# %%
from dataset.cifar_dataset import Cifar10Dataset, Cifar100Dataset
from dataset.mnist_dataset import MnistDataset
from utils.transformer import ImgTransform, BboxTransform, LabelTransform

class RepeatDataset(object):

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
#        self.CLASSES = dataset.CLASSES
#        if hasattr(self.dataset, 'flag'):
#            self.flag = np.tile(self.dataset.flag, times)
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len
    
    
def get_dataset(dataset_cfg, transform_cfg):
    """创建数据集
    """
    datasets = {'cifar10' : Cifar10Dataset,
                'cifar100' : Cifar100Dataset,
                'mnist' : MnistDataset}
    img_transform = None
    label_transform = None
    bbox_transform = None    
    
    if transform_cfg.get('img_params') is not None:
        img_p = transform_cfg['img_params']
        img_transform = ImgTransform(**img_p)
    if transform_cfg.get('label_params') is not None:
        label_p = transform_cfg['label_params']
        label_transform = LabelTransform(**label_p)
    if transform_cfg.get('bbox_params') is not None:
        bbox_p = transform_cfg['bbox_params']
        bbox_transform = BboxTransform(**bbox_p)
    
    dataset_name = dataset_cfg.get('type')
    dataset_class = datasets[dataset_name]
    params = dataset_cfg.get('params', None)
    repeat = dataset_cfg.get('repeat', 0)
    
    dset = dataset_class(**params, 
                         img_transform=img_transform,
                         label_transform=label_transform,
                         bbox_transform=bbox_transform)    
    if repeat:
        return RepeatDataset(dset, repeat)
    else:
        return dset


# %%
from torch.utils.data.dataloader import default_collate
def multi_collate(batch):
    """自定义一个多数据分别堆叠的collate函数：此时要求数据集输出为[img,label, scale,..]
    参考：https://www.jianshu.com/p/bb90bff9f6e5
    原有default_collate在处理多种不同类型数据时，输出有时不是所需形式。通过自定义
    multi_collate函数，可实现tensor/numpy/inf/float各种结构的堆叠，生成符合batch_data需求的数据。
    数据集输出：list
    输入：batch为list(list)，来自dataloader，[[img1,label1...],[img2,label2,...],[img3,label3,...]]
    输出：result为list，每个变量都是堆叠好的，[img, label, scale]
    """
    result = []
    for sample in zip(*batch):  # 参考pytorch源码写法：数据先分组提取
        if isinstance(sample[0], torch.Tensor):
            stacked = torch.stack(sample, dim=0)    
            result.append(stacked)
        if isinstance(sample[0], np.ndarray):
            stacked = np.stack(sample, axis=0)
            result.append(torch.tensor(stacked))
        if isinstance(sample[0], (tuple,list)):
            stacked = np.stack(sample, axis=0)
            result.append(torch.tensor(stacked))
        if isinstance(sample[0], (int, float)):
            stacked = np.stack(sample, axis=0)
            result.append(torch.tensor(stacked))
    return result
    

def dict_collate(batch):
    """自定义字典堆叠式collate，此时要求数据集输出为一个dict。
    数据集输出：dict
    输入：batch为list(dict), 每个元素为一个dict, [{'img':img, 'label', label, 'scale':scale}, {..}, {..}]
    输出：result为dict, 每个value都是堆叠好的 {'img':img, 'label':label, 'scale':scale,...}
    """
    result = {}
    data = batch[0].values()
    data = list(data)  # 取出数据用于检查类型
    for i, name in enumerate(batch[0].keys()):  # 第i个变量的堆叠
        if isinstance(data[i], torch.Tensor):  
            stacked = torch.stack([sample[name] for sample in batch])
            result[name] = stacked
        if isinstance(data[i], np.ndarray):
            stacked = np.stack([sample[name] for sample in batch])
            result[name] = torch.tensor(stacked)
        if isinstance(data[i], (tuple,list)):
            stacked = np.stack([sample[name] for sample in batch])
            result[name] = torch.tensor(stacked)
        if isinstance(data[i], (int, float)):
            stacked = np.stack([sample[name] for sample in batch])
            result[name] = torch.tensor(stacked)
    return result  # 期望的result应该是{'img': img, 'label':label}
    

def get_dataloader(dataset, dataloader_cfg):
    """生成pytorch版本的dataloader，在cfg中只提供基础参数，而那些需要自定义函数的则需要另外传入。
    相关参数包括:
    Dataloader(dataset, batch_size, shuffle=False, sampler=None, batch_sampler=None,
               num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
               timeout=0, worker_init_fn=None)    
    
    参考：https://www.jianshu.com/p/bb90bff9f6e5
    1. sampler: (可自定义函数)用来生成一系列index, 
       pytorch已经实施的sampler包括SequentialSampler, RandomSampler, WeightedSampler, SubstRandomSampler.
       当shuffle=True时默认调用RandomSampler, 而当shuffle=False时默认调用SequentialSampler.
       如果自定义了sampler，则需要设置shuffle=False
    2. batch_sampler: 用来把sampler生成的index打包分组，得到多个batch的index.
       如果shuffle=True，默认的sampler是RandomSampler, 默认的batch_sampler为pytorch已实现的。
       如果shuffle=False，默认的sampler是SequentialSampler，默认的batch_sampler为pytorch已实现的。
    3. collate_fn：(可自定义函数)用来把一个batch的数据进行合并，比如多张img合并成一个数据，多个labels合并成一个数据。
       默认为default_collate,可通过from torch.utils.data.dataloader import default_collate来导入,
       它实现的功能核心就是torch.stack(batch, 0, out=out)也就是通过stack升维堆叠从(c,h,w)变成(b,c,h,w)
       因为常规__getitem__配合默认collate_fn只返回img,label，但如果想要返回img, box, label等信息，
       就需要自定义collate_fn来合并成一个batch，方便后续的训练。
    
    注意：
    1. 如果数据集getitem输出的是img(3,32,32),label(标量tensor(1))，则经过dataloader输出会堆叠为img(64,3,32,32)的4维度数组, label([64])的1维度数组)
       而当数据集getitem输出的是img, label, scale, shape，
    
    """
    collate_fn_dict = {'default_collate':default_collate,
                       'multi_collate':multi_collate,
                       'dict_collate':dict_collate}
    sampler_fn_dict = {}
    sampler = None
    collate_fn = default_collate   # 注意不能让collate_fn的默认参数为None,否则会导致不可调用的报错
    
    params = dataloader_cfg.get('params', None)
    c_name = params.pop('collate_fn')
    if c_name is not None:  # 创建自定义collate_fn
        collate_fn = collate_fn_dict[c_name]
    
    s_name = params.pop('sampler')
    if s_name is not None:    # 创建自定义sampler
        sampler = sampler_fn_dict[s_name]
    
    return torch.utils.data.DataLoader(dataset, **params, 
                                       sampler=sampler, collate_fn=collate_fn)
        

# %%
from model.cnn_detector_lib import OneStageDetector
from model.cnn_alexnet_lib import AlexNet8
from model.cnn_ssdvgg16_lib import SSDVGG16
from model.cnn_head_lib import SSDHead

def get_model(model_cfg):
    """创建模型：如果创建集成模型(detector)，则需要传入根cfg，如果创建单模型，则需要传入该模型cfg_model
    """
    models = {
            'one_stage_detector': OneStageDetector,
            'alexnet8' : AlexNet8,
            'vgg16' : SSDVGG16,
            'ssdhead' : SSDHead}
    
    if model_cfg.get('type', None) is None and model_cfg.task=='detector':   # 不包含type的detector
        model_name = model_cfg.model['type']
        model_class = models[model_name]
        model = model_class(model_cfg)        # 不包含type的classifier
        return model

    elif model_cfg.get('type', None) is None and model_cfg.task=='classifier':         
        model_name = model_cfg.model.get('type')
        model_class = models[model_name]
        params = model_cfg.model.params
        return model_class(**params)
            
    elif model_cfg.get('type', None) is not None:   # 否则就是直接创建model
        model_name = model_cfg['type']
        model_class = models[model_name]
        params = model_cfg.params    
        return model_class(**params)  # 其他模型的创建，传入的是解包的dict


# %%       
def get_optimizer(optimizer_cfg, model):
    """创建优化器：pytorch采用的单个优化器更新所有权重数据，所以需要传入给优化器model的所有权重数据
    """
    optimizers = {
            'sgd' : torch.optim.SGD,
            'adam' : torch.optim.Adam,
            'rms_prop' : torch.optim.Rprop}
    
    opt_name = optimizer_cfg.get('type')
    opt_class = optimizers[opt_name]
    params = optimizer_cfg['params']
    model_params = dict(params=model.parameters())  # 获取模型所有权重
    for name, value in model_params.items():
        params.setdefault(name, value)
    return opt_class(**params)    # 把模型权重，以及自定义的lr/momentum/weight_decay一起传入optimizer


# %%
def get_loss_fn(loss_cfg):
    loss_fn_dict = {
            'cross_entropy': torch.nn.CrossEntropyLoss,
            'smooth_l1': torch.nn.SmoothL1Loss}
    
    loss_name = loss_cfg.get('type')
    loss_class = loss_fn_dict[loss_name]
    params = loss_cfg.get('params')
    if params is not None:  # 带超参损失函数
        return loss_class(**params)
    else:                    # 不带超参损失函数
        return loss_class()



# %%
from model.lr_processor_lib import FixedLrProcessor, ListLrProcessor, StepLrProcessor
        
def get_lr_processor(runner, lr_processor_cfg):
    lr_processors = {'fix': FixedLrProcessor,
                    'list': ListLrProcessor,
                    'step': StepLrProcessor}
    lr_processor_name = lr_processor_cfg.type
    lr_processor_class = lr_processors[lr_processor_name]
    params = lr_processor_cfg.params
    params.setdefault('runner', runner)
    return lr_processor_class(**params)
    
    
        
        


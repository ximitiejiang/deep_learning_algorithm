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

from dataset.cifar_dataset import Cifar10Dataset, Cifar100Dataset
from dataset.voc_dataset import VOCDataset
from dataset.ants_bees_dataset import AntsBeesDataset
from dataset.widerface_dataset import WIDERFaceDataset
from utils.transform import ImgTransform, BboxTransform, LabelTransform, SegTransform, MaskTransform

from model.detector_lib import OneStageDetector, Segmentator
from model.alexnet_lib import AlexNet, AlexNet8
from model.ssdvgg16_lib import SSDVGG16
from model.fcnvgg16_lib import FCNVGG16
from model.head_lib import SSDHead, RetinaHead, FCOSHead
from model.fcn_head import FCN8sHead


# %% model zoo
datasets = {'cifar10' : Cifar10Dataset,
            'cifar100' : Cifar100Dataset,
            'voc' : VOCDataset,
            'antsbees': AntsBeesDataset,
            'widerface': WIDERFaceDataset}

models = {
        'one_stage_detector': OneStageDetector,
        'segmentator' : Segmentator,
        'alexnet8' : AlexNet8,
        'alexnet' : AlexNet,
        'ssd_vgg16' : SSDVGG16,
        'fcn_vgg16' : FCNVGG16,
        'ssd_head' : SSDHead,
        'retina_head': RetinaHead,
        'fcos_head': FCOSHead,
        'fcn8s_head': FCN8sHead}

loss_fn_dict = {
            'cross_entropy': torch.nn.CrossEntropyLoss,
            'smooth_l1': torch.nn.SmoothL1Loss}


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

    img_transform = None
    label_transform = None
    bbox_transform = None
    aug_transform = None
    mask_transform = None
    seg_transform = None
    
    if transform_cfg.get('img_params') is not None:
        img_p = transform_cfg['img_params']
        img_transform = ImgTransform(**img_p)
    if transform_cfg.get('label_params') is not None:    
        label_p = transform_cfg['label_params']
        label_transform = LabelTransform(**label_p)
    if transform_cfg.get('bbox_params') is not None:        
        # 且由于bbox_transform比较特殊，大部分变换参数取决于img_transform，因此在call的时候输入
        bbox_p = transform_cfg['bbox_params']
        bbox_transform = BboxTransform(**bbox_p)
    if transform_cfg.get('seg_params') is not None:
        seg_p = transform_cfg['seg_params']
        seg_transform = SegTransform(**seg_p)
    if transform_cfg.get('mask_params') is not None:
        mask_p = transform_cfg['mask_params']
        mask_transform = MaskTransform(**mask_p)
        
    dataset_name = dataset_cfg.get('type')
    dataset_class = datasets[dataset_name]
    params = dataset_cfg.get('params', None)
    repeat = dataset_cfg.get('repeat', 0)
    
    dset = dataset_class(**params, 
                         img_transform=img_transform,
                         label_transform=label_transform,
                         bbox_transform=bbox_transform,
                         aug_transform=aug_transform,
                         seg_transform=seg_transform,
                         mask_transform=mask_transform)    
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
    需要解决2个问题：1.如果一个batch的每张img尺寸都不同如何堆叠？2.如果meta不是数组无法堆叠如何处理？
    问题1的解决方案是通过pad成最大尺寸后再进行堆叠，这里不能做成列表因为后续神经网络的前向计算是需要堆叠后的数据进行操作。
        在每个dict中增加一个stack关键字'stack': ['img', 'segment']用来指定哪些变量需要堆叠。堆叠方式就是先填充成最大图片然后再堆叠。
    问题2的解决方案是做成列表而不进行堆叠。
    数据集输出：dict
    输入：batch为list(dict), 每个元素为一个dict, [{'img':img, 'label', label, 'scale':scale}, {..}, {..}]
    输出：result为dict(list), 每个value都分类放在同一list中 {'img':[imgs], 'label':[labels], 'scale':[scales],...}
    """
    result = {}
    stack_list = batch[0]['stack_list']
    for i, name in enumerate(batch[0].keys()):  # 第i个变量的堆叠
        if name in stack_list:  # 如果需要堆叠
            data_list = [sample[name] for sample in batch]
            shapes = np.stack([data.shape for data in data_list], axis=0) # (k, 3) or (k, 2)
            # 如果是3维img的堆叠
            if len(shapes[0]) == 3:   
                max_c, max_h, max_w = np.max(shapes, axis=0)
                stacked = data_list[0].new_zeros(len(batch), max_c, max_h, max_w)  # b,c,h,w
                for dim in range(len(batch)):
                    data = data_list[dim]
                    stacked[dim,:data.shape[0], :data.shape[1], :data.shape[2]] = data #
            # 如果是二维seg的堆叠
            elif len(shapes[0]) == 2: 
                max_h, max_w = np.max(shapes, axis=0)
                stacked = data_list[0].new_zeros(len(batch), max_h, max_w)  # b, h, w
                for dim in range(len(batch)):
                    data = data_list[dim]
                    stacked[dim, :data.shape[0], :data.shape[1]] = data
                    
            result[name] = stacked
        else:  # 如果不需要堆叠: 则放入一个list,即 [tensor1, tensor2..]
            result[name] = [sample[name] for sample in batch]
    return result
    

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

def get_root_model(cfg):
    """根模型创建：传入根cfg"""
    # 如果是classifier单模型的根模型
    if cfg.get('backbone', None) is None:
        return get_model(cfg.model)
    # 如果是detector复合模型，送入根cfg
    elif cfg.get('backbone', None) is not None:
        model_name = cfg.model['type']
        model_class = models[model_name]
        model = model_class(cfg)
        return model

def get_model(cfg):
    """创建单模型：传入模型cfg
    """
    model_name = cfg['type']
    model_class = models[model_name]
    params = cfg.params    
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


# %%
#from torch.utils.tensorboard import SummaryWriter
#
#class TensorBoardWriter():
#    """创建tensorboard的writer类
#    参考：https://pytorch.org/docs/stable/tensorboard.html
#    """
#    def __init__(self):
#        self.writer = SummaryWriter()
#    
#    def update(self, data=None, grid=None, title='result'):
#        if grid is not None:
#            self.writer.add_image(title, grid, 0)
#        if data is not None:
#            self.writer.add_scalar(title, data)
    

if __name__ == "__main__":
    pass    
        
        


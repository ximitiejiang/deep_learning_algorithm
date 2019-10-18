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
from torch.nn.parallel import DistributedDataParallel

from utils.transform import ImgTransform, BboxTransform, LabelTransform
from utils.transform import SegTransform, AugTransform, LandmarkTransform
from utils.tools import get_time_str

from dataset.cifar_dataset import Cifar10Dataset, Cifar100Dataset
from dataset.voc_dataset import VOCDataset
from dataset.coco_dataset import CocoDataset
from dataset.ants_bees_dataset import AntsBeesDataset
from dataset.widerface_dataset import WIDERFaceDataset
from dataset.cityscapes_dataset import CityScapesDataset

from model.assembly_model_lib import OneStageDetector, Segmentator, Classifier
from model.backbone.alexnet_lib import AlexNet, AlexNet8
from model.backbone.resnet_lib import ResNet
from model.backbone.ssdvgg16_lib import SSDVGG16
from model.backbone.fcnvgg16_lib import FCNVGG16
from model.neck.neck_lib import FPN
from model.bbox_head.ssd_head import SSDHead
from model.bbox_head.fcos_head import FCOSHead
from model.seg_head.fcn_head import FCN8sHead

from model.lr_processor_lib import FixedLrProcessor, ListLrProcessor, StepLrProcessor


# %% model zoo
datasets = {'cifar10' : Cifar10Dataset,
            'cifar100' : Cifar100Dataset,
            'voc' : VOCDataset,
            'coco': CocoDataset,
            'cityscapes' : CityScapesDataset, 
            'antsbees': AntsBeesDataset,
            'widerface': WIDERFaceDataset}

models = {
        'one_stage_detector': OneStageDetector,
        'segmentator' : Segmentator,
        'classifier' : Classifier,
        'alexnet8' : AlexNet8,
        'alexnet' : AlexNet,
        'resnet'  : ResNet,
        'ssd_vgg16' : SSDVGG16,
        'fcn_vgg16' : FCNVGG16,
        'fpn' : FPN,
        'ssd_head' : SSDHead,
        'fcos_head': FCOSHead,
        'fcn8s_head': FCN8sHead}

lr_processors = {'fix': FixedLrProcessor,
                'list': ListLrProcessor,
                'step': StepLrProcessor}

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



def merge_config(args, cfg):
    """把args合并到cfg, 可采用vars()把namespace转换成dict
    其中cfg为dict, args为namespace
    """
    args = vars(args)
    for key, value in args.items():
        if value is not None:
            cfg[key] = value
    return cfg


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
def get_device(cfg, logger=None):
    """生成训练对应的设备：如果是分布式，则不同local rank(不同进程号)返回的是不同设备"""
    # 如果是分布式设备: 当前默认是采用了所有GPU进行分布计算
    # TODO: 分布训练下能够指定具体gpu id
    if cfg.distribute and len(cfg.gpus) > 1 and torch.cuda.is_available():
        local_rank = int(os.environ['RANK'])  # 获得该进程的进程号
        num_gpus = torch.cuda.device_count()  # 获得本机总的GPU数
        device_id = local_rank % num_gpus     # 定义device_id: 没有直接用local_rank做device_id是因为这样可以设置只使用所有gpu中的其中一部分。
        device = torch.cuda.device(device_id)
        info = 'Operation will start in distributed GPUs.'
    # 如果是并行式单机设备: 提示用分布式，因为分布式比并行式更快。
    elif not cfg.distribute and len(cfg.gpus) > 1 and torch.cuda.is_available(): 
        device = torch.cuda.device(cfg.gpus[0])   # 并行式的设备只用第一个GPU作为主GPU
        info = 'Operation will start in parallel GPUs.'
    # 如果是单GPU设备    
    elif len(cfg.gpus) == 1 and torch.cuda.is_available():
        device = torch.device("cuda")   # 设置设备GPU: "cuda"和"cuda:0"的区别？
        info = 'Operation will start in one GPU!'
    # 如果是cpu设备
    elif cfg.gpus is None:
        device = torch.device("cpu")      # 设置设备CPU
        info = 'Operation will start in CPU!'
    else:
        raise ValueError('Can not get correct device from get_device().')
    if logger is not None:
        logger.info(info)
    return device


# %%

class RepeatDataset(object):

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
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
    landmark_transform = None
    seg_transform = None
    
    if transform_cfg.get('img_params', None) is not None:
        img_p = transform_cfg['img_params']
        img_transform = ImgTransform(**img_p)

    if transform_cfg.get('label_params', None) is not None:    
        label_p = transform_cfg['label_params']
        label_transform = LabelTransform(**label_p)

    if transform_cfg.get('bbox_params', None) is not None:        
        # 且由于bbox_transform比较特殊，大部分变换参数取决于img_transform，因此在call的时候输入
        bbox_p = transform_cfg['bbox_params']
        bbox_transform = BboxTransform(**bbox_p)

    if transform_cfg.get('landmark_params', None) is not None:
        landmark_p = transform_cfg['landmark_params']
        landmark_transform = LandmarkTransform(**landmark_p)
        
    if transform_cfg.get('seg_params', None) is not None:
        seg_p = transform_cfg['seg_params']
        seg_transform = SegTransform(**seg_p)

    if transform_cfg.get('aug_params', None) is not None:
        aug_p = transform_cfg['aug_params']
        aug_transform = AugTransform(**aug_p)
        
    dataset_name = dataset_cfg.get('type')
    dataset_class = datasets[dataset_name]
    params = dataset_cfg.get('params', None)
    repeat = dataset_cfg.get('repeat', 0)
    
    dset = dataset_class(**params, 
                         img_transform=img_transform,
                         label_transform=label_transform,
                         bbox_transform=bbox_transform,
                         aug_transform=aug_transform,
                         landmark_transform=landmark_transform,
                         seg_transform=seg_transform)    
    if repeat:
        return RepeatDataset(dset, repeat)
    else:
        return dset


# %%
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DistributedSampler
from utils.tools import get_dist_info

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
    

class NewDistributedSampler(DistributedSampler):
    """分布式采样：在pytorch原始分布式采样基础上微调，由于pytorch原版分布式采样只能随机采样，无法设置是否shuffle, 也就不能顺序采样。
    增加shuffle参数，从而让单机版和分布式都可以基于cfg设置shuffle与否，在分布式预测时就可以获得顺序采样的效果。
    sampler逻辑：重写__iter__()函数，返回iter(一个序列号列表)
    
    分布式采样逻辑：根据当前rank号，来提取一组样本序列号。比如rank0的样本序列号indices=shuffled_dataset_indices[0:len:n_rank], len表示整个数据集长度，n_rank表示多少块GPU。
    从而通过控制起始的idx号和步长n_rank很巧妙的把一组indices分成多组。
    
    输入：dataset数据集
    输出：整个数据集的索引号idx
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # 默认的shuffle采样
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        # 新增部分：如果不shuffle，则顺序采样
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)



def get_dataloader(dataset, dataloader_cfg, num_gpus, dist=False):
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
    collate_fn_dict = {'default' : default_collate,
                       'multi_collate' : multi_collate,
                       'dict_collate' : dict_collate}
    sampler_fn_dict = {'default' : DistributedSampler,
                       'group_sampler': None}
    # 注意不能让collate_fn的默认参数为None,否则会导致不可调用的报错
    collate_fn = default_collate   
    params = dataloader_cfg.get('params', None)
    
    # 定义collate_fn
    collate_name = params.pop('collate_fn')
    if collate_name is not None:  # 创建自定义collate_fn
        collate_fn = collate_fn_dict[collate_name]
    
    # 定义sampler
    sampler_name = params.pop('sampler')
    pin_memory = params.get('pin_momory', False)
    drop_last = params.get('drop_last', False)
    if not dist:
        sampler = None   # 非dist条件下设为None，但实际上内部sampler取决于shuffle的定义，如果shuffle=True则自动为RandomSampler，否则为SequentialSampler
        if num_gpus > 0: # GPU
            shuffle = params['shuffle']
            batch_size = params['batch_size'] * num_gpus       # GPU模式下(单GPU或并行)则batch size要乘以GPU块数，cpu模式则不变
            num_workers = params['num_workers'] * num_gpus
        else:  # CPU
            shuffle = params['shuffle']
            batch_size = params['batch_size']       # GPU模式下则batch size要乘以GPU块数，cpu模式则不变
            num_workers = params['num_workers']            

    # pytorch默认的分布式采样模块
    elif (dist and sampler_name=='default') or (dist and sampler_name is None):
        rank, world_size = get_dist_info()
        sampler = sampler_fn_dict['default'](dataset, world_size, rank)  # 创建分布式采样器：数据集，GPU块数，等级
        shuffle = False   # 在分布式情况下，不需要设置随机打乱，因为分布式采样模块本身就是随机的
        batch_size = params['batch_size']
        num_workers = params['num_workers']
    # 生成dataloader
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             shuffle=shuffle,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             sampler=sampler, 
                                             collate_fn=collate_fn)
    return dataloader


# %%
import torch.nn as nn
def get_model(cfg):
    """创建单模型：传入模型cfg
    """
    # 如果传入的是整个cfg，则初始化总成模型
    if cfg.get('model', None) is not None:
        model_name = cfg.model['type']
        model_class = models[model_name]
        model = model_class(cfg)
    # 如果传入的是某个model的cfg，则初始化单模型
    else:
        model_name = cfg['type']
        model_class = models[model_name]
        params = cfg.params    
        model = model_class(**params)  # 其他模型的创建，传入的是解包的dict
    return model
        

def get_model_wrapper(model, cfg):
    """为模型添加外壳：生成并行式模型或者分布式模型"""
    if cfg.gpus is None:
        return model

    # 并行式模型
    if cfg.parallel and len(cfg.gpus) > 1 and torch.cuda.device_count() > 1:  # 判断并行式，gpu个数
        model = nn.DataParallel(model)
    # 总成模型判断是否分布式
    elif cfg.distribute and len(cfg.gpus) > 1 and os.environ.get('RANK', None) is not None:  # 判断分布式，gpu个数，且dist启动
         local_rank = os.environ['RANK']
         model = DistributedDataParallel(model, 
                                        device_ids=[local_rank],   # 模型所在的进程号：说明模型会送入相应
                                        output_device=local_rank)  # 模型
    return model

# %%       
def get_optimizer(optimizer_cfg, model):
    """创建优化器：pytorch采用的单个优化器更新所有权重数据，所以需要传入给优化器model的所有权重数据
    """
    optimizers = {
            'sgd' : torch.optim.SGD,
            'adam' : torch.optim.Adam,
            'rms_prop' : torch.optim.Rprop}
    if hasattr(model, 'module'):  # 如果是并行模型，为了获得参数需要先去掉module外壳。
        model = model.module
    
    opt_name = optimizer_cfg.get('type')
    opt_class = optimizers[opt_name]
    params = optimizer_cfg['params']
    model_params = dict(params=model.parameters())  # 获取模型所有权重
    for name, value in model_params.items():
        params.setdefault(name, value)
    return opt_class(**params)    # 把模型权重，以及自定义的lr/momentum/weight_decay一起传入optimizer


# %%
def get_lr_processor(runner, lr_processor_cfg):

    lr_processor_name = lr_processor_cfg.type
    lr_processor_class = lr_processors[lr_processor_name]
    params = lr_processor_cfg.params
    params.setdefault('runner', runner)
    return lr_processor_class(**params)


# %%
    
# TODO: 暂时不开发，因为必须升级pytorch版本到1.14，当前pytorch是1.1
#from torch.utils.tensorboard import SummaryWriter
#
#class TensorBoardWriter():
#    """创建tensorboard的writer类
#    参考：https://pytorch.org/docs/stable/tensorboard.html
#    用法：
#    writer = SummaryWriter()
#    https://localhost:6006
#        1. 显示图片：一个batch的图片(b,c,h,w)被grid后，就可提供到writer去显示
#            img_grid = torchvision.utils.make_grid(images)  
#            writer.add_image('batch img', img_grid)
#        2. 显示模型
#            writer.add_graph(model, images)
#        3. 显示曲线
#            writer.add_scalar('loss', ruuning_loss / 1000, epoch)
#    """
#    def __init__(self):
#        self.writer = SummaryWriter()  # TODO: 是否需要额外增加一个记录summerywriter的dir，跟logger的dir并列？
#    
#    def update(self, record):
#        if isinstance(record, dict):
#            for key, value in record.items():
#                self.writer.add_scalar()
#        if isinstance(record, torch.Tensor):
#            pass
#        
##        if grid is not None:
##            self.writer.add_image(title, grid, 0)
##        if data is not None:
##            self.writer.add_scalar(title, data)
#    
#    def close(self):
#        self.writer.close()

    

if __name__ == "__main__":
    pass    
        
        


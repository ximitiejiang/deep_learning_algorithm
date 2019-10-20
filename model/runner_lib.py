#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:46:38 2019

@author: ubuntu
"""
import torch
import torch.nn as nn
import os
import time

from utils.prepare_training import get_logger, get_dataset, get_dataloader, get_device 
from utils.prepare_training import get_model, get_model_wrapper, get_optimizer, get_lr_processor
from utils.visualization import vis_loss_acc
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.transform import to_device
    

def batch_detector(model, data, device, return_loss=True, **kwargs): # kwargs用来兼容分类时传入的loss_fn
    # 数据送入设备：注意数据格式问题改为在to_tensor中完成，数据设备问题改为在to_device完成
    imgs = to_device(data['img'], device)
    gt_bboxes = to_device(data['gt_bboxes'], device)
    gt_labels = to_device(data['gt_labels'], device)
    if data.get('gt_landmarks', None) is not None:
        gt_landmarks = to_device(data['gt_landmarks'], device)
    else:
        gt_landmarks = None
    img_metas = data['img_meta']
    # 计算模型输出
    if not return_loss:
        bbox_det = model(imgs, img_metas, return_loss=False)
        return bbox_det  # (n_class,)(k,5)
        
    if return_loss:
        losses = model(imgs, img_metas, 
                       gt_bboxes=gt_bboxes, 
                       gt_labels=gt_labels, 
                       gt_landmarks=gt_landmarks, return_loss=True)  # 
        # 损失缩减：先分别对每种loss进行batch内的求和，并对不同种loss进行求和。
        loss_sum = {}
        for name, value in zip(losses.keys(), losses.values()):
            loss_sum[name] = sum(data for data in value)
        loss = sum(data for data in loss_sum.values())
        outputs = dict(loss=loss)
        
    return outputs


def batch_segmentator(model, data, device, return_loss=True, **kwargs):
    imgs = to_device(data['img'], device)
    segs = to_device(data['seg'], device)
    if not return_loss:
        outs = model(imgs, return_loss=False, segs=segs)
        return outs
    if return_loss:
        loss = model(imgs, return_loss=True, segs=segs)
        return loss
    
    

def batch_classifier(model, data, device, return_loss=True, **kwargs):
    # 数据送入设备
    img = to_device(data['img'], device)  
    label = to_device(data['gt_labels'], device)
    if return_loss:
        outs = model(imgs=img, return_loss=True, labels=label)  # pytorch交叉熵包含了前端的softmax/one_hot以及后端的mean
        return outs   # 包括loss, acc
    else:
        outs = model(imgs=img, return_loss=False)
        return outs   # tbd


def get_batch_processor(cfg):
    if cfg.model.type == 'classifier':
        return batch_classifier
    elif cfg.model.type == 'one_stage_detector':
        return batch_detector
    elif cfg.model.type == 'segmentator':
        return batch_segmentator
    else:
        raise ValueError('Wrong task input.')


class Runner():
    """创建一个runner类，用于服务pytorch中的模型训练和验证: 支持cpu/单gpu/多gpu并行训练/分布式训练
    Runner类用于操作一个主模型，该主模型可以是单个分类模型，也可以是一个集成检测模型。
    所操作的主模型需要继承自pytorch的nn.module，且包含forward函数进行前向计算。
    如果主模型是一个集成检测模型，则要求该主模型所包含的所有下级模型也要继承
    自nn.module且包含forward函数。
    """
    def __init__(self, cfg, 
                 resume_from=None):
        # 共享变量: 需要声明在resume/load之前，否则会把resume的东西覆盖
        self.c_epoch = 0
        self.c_iter = 0
        self.weight_ready = False
        self.buffer = {'loss': [],
                       'acc1': [],
                       'acc5': [],
                       'lr':[]}
        # 获得配置
        self.cfg = cfg
        if resume_from is not None:
            self.cfg.resume_from = resume_from  # runner可以直接修改resume_from,避免修改cfg文件
        # 检查文件夹和文件是否合法
        self.check_dir_file(self.cfg)
        #设置logger
        self.logger = get_logger(self.cfg.logger)
        self.logger.info('start logging info.')
        #设置设备: 如果是分布式，则不同local rank(不同进程号)返回的是不同设备
        self.device = get_device(self.cfg, self.logger)
        #创建batch处理器
        self.batch_processor = get_batch_processor(self.cfg)
        #创建数据集
        self.trainset = get_dataset(self.cfg.trainset, self.cfg.transform)
        self.valset = get_dataset(self.cfg.valset, self.cfg.transform_val) # 做验证的变换只做基础变换，不做数据增强
        
#        tmp1 = self.trainset[91]  # for debug: 可查看dataset __getitem__
        
        #创建数据加载器
        self.dataloader = get_dataloader(self.trainset, 
                                         self.cfg.trainloader, 
                                         len(self.cfg.gpus), 
                                         dist=self.cfg.distribute)
        self.valloader = get_dataloader(self.valset, 
                                        self.cfg.valloader,
                                        len(self.cfg.gpus),
                                        dist=self.cfg.distribute)
        
#        tmp2 = next(iter(self.dataloader))  # for debug: 设置worker=0就可查看collate_fn
        
        # 创建模型并初始化
        if self.cfg.load_from is not None or self.cfg.resume_from is not None:
            self.cfg.backbone.params.pretrained = None  # 如果load_from或resume_from，则不加载pretrained
        self.model = get_model(self.cfg)
        
        # 优化器：必须在model送入cuda之前创建
        self.optimizer = get_optimizer(self.cfg.optimizer, self.model)
        # 学习率调整器
        self.lr_processor = get_lr_processor(self, self.cfg.lr_processor)
        # 送入GPU
        # 包装并行模型是在optimizer提取参数之后，否则可能导致无法提取，因为并行模型在model之下加了一层module壳
        self.model = get_model_wrapper(self.model, self.cfg)
        self.model.to(self.device)
        # 注意：恢复或加载是直接加载到目标设备，所以必须在模型传入设备之后进行，确保设备匹配
        # 加载模型权重和训练参数，从之前断开的位置继续训练
        if self.cfg.resume_from:
            self.resume_training(checkpoint_path=self.cfg.resume_from, 
                                 map_location=self.device)  # 沿用设置的device
        # 加载模型权重，但没有训练参数，所以一般用来做预测
        elif self.cfg.load_from:
            load_device = torch.device(self.cfg.load_device)
            self._load_checkpoint(checkpoint_path=self.cfg.load_from, 
                                  map_location=load_device)
            self.weight_ready = True
      
    def check_dir_file(self, cfg):
        """检查目录和文件的合法性，防止运行中段报错导致训练无效"""
        # 检查文件合法性
        if cfg.get('resume_from', None) is not None:
            file = cfg.resume_from
            if not os.path.isfile(file):
                raise FileNotFoundError('resume_from file is not a file.')
        if cfg.get('load_from', None) is not None:
            file = cfg.load_from
            if not os.path.isfile(file):
                raise FileNotFoundError('load_from file is not a file.')
        # 检查路径合法性
        if cfg.get('work_dir', None) is not None:
            dir = cfg.work_dir
            if not os.path.isdir(dir):
                raise FileNotFoundError('work_dir is not a dir.')
        if cfg.get('data_root_path', None) is not None:
            dir = cfg.trainset.params.root_path
            if not os.path.isdir(dir):
                raise FileNotFoundError('trainset path is not a dir.')
    
    def current_lr(self):
        """获取当前学习率: 其中optimizer.param_groups有可能包含多个groups(但在我的应用中只有一个group)
        也就是为一个单元素list, 取出就是一个dict.
        即param_groups[0].keys()就包括'params', 'lr', 'momentum','dampening','weight_decay'
        返回：
            current_lr(list): [value0, value1,..], 本模型基本都是1个value，除非optimizer初始化传入多组lr
        """
        return [group['lr'] for group in self.optimizer.param_groups]  # 取出每个group的lr返回list，大多数情况下，只有一个group
    
    def train(self, vis=False):
        """用于模型在训练集上训练"""
        self.model.train() # module的通用方法，可自动把training标志位设置为True
        self.lr_processor.set_base_lr_group()  # 设置初始学习率(直接从optimizer读取到：所以save model时必须保存optimizer) 
        start = time.time()
        while self.c_epoch < self.cfg.n_epochs:
            self.lr_processor.set_regular_lr_group()  # 设置常规学习率(计算出来并填入optimizer)
            for self.c_iter, data_batch in enumerate(self.dataloader):
                self.lr_processor.set_warmup_lr_group() # 设置热身学习率(计算出来并填入optimizer)
                # 前向计算
                outputs = self.batch_processor(self.model, 
                                               data_batch, 
                                               self.device,
                                               return_loss=True)
                # 反向传播: 注意随时检查梯度是否爆炸
                outputs['loss'].backward()  # 更新反向传播, 用数值loss进行backward()      
                self.optimizer.step()   
                self.optimizer.zero_grad()       # 每个batch的梯度清零
                # 存放结果
                self.buffer['loss'].append(outputs.get('loss', torch.tensor(0.)))
                self.buffer['acc1'].append(outputs.get('acc1', torch.tensor(0.)))
                self.buffer['acc5'].append(outputs.get('acc5', torch.tensor(0.)))
                self.buffer['lr'].append(self.current_lr()[0])
                # 显示text
                if (self.c_iter+1)%self.cfg.logger.interval == 0:
                    lr_str = ','.join(['{:.4f}'.format(lr) for lr in self.current_lr()]) # 用逗号串联学习率得到一个字符串
                    log_str = 'Epoch [{}][{}/{}]\tloss: {:.4f}, acc1: {:.4f}, acc5: {:.4f}\tlr: {}'.format(self.c_epoch+1, 
                                     self.c_iter+1, len(self.dataloader), 
                                     self.buffer['loss'][-1].item(),
                                     self.buffer['acc1'][-1].item(), 
                                     self.buffer['acc5'][-1].item(), lr_str)

                    self.logger.info(log_str)
            
            # 保存模型
            if self.c_epoch%self.cfg.save_checkpoint_interval == 0:
                self.save_training(self.cfg.work_dir)            
            # 如果不需要显示loss/acc，则清除buffer
            if not vis:
                for key in self.buffer.keys():
                    self.buffer[key] = [] 
            self.c_epoch += 1
        times = time.time() - start
        self.logger.info('training finished with times(s): {}'.format(times))
        # 绘图
        if vis:
            vis_loss_acc(self.buffer, title='train')
        self.weight_ready= True
    
    
    def val(self, vis=True):
        """用于模型验证"""
        self.buffer = {'acc': []}  # 重新初始化buffer,否则acc会继续累加
        self.n_correct = 0    # 用于计算全局acc
        if self.weight_ready:
            self.model.eval()   # 关闭对batchnorm/dropout的影响，不再需要手动传入training标志
            for c_iter, data_batch in enumerate(self.valloader):
                with torch.no_grad():  # 停止反向传播，只进行前向计算
                    outputs = self.batch_processor(self.model, 
                                                   data_batch, 
                                                   self.device,
                                                   return_loss=False)
                    self.buffer['acc'].append(outputs['acc1'])
                # 计算总体精度
                self.n_correct += self.buffer['acc'][-1] * len(data_batch['gt_labels'])
            
            if vis:
                vis_loss_acc(self.buffer, title='val')
            self.logger.info('ACC on valset: %.3f', self.n_correct/len(self.valset))
        else:
            raise ValueError('no model weights loaded.')
    
    
    def save_training(self, out_dir):
        meta = dict(c_epoch = self.c_epoch,
                    c_iter = self.c_iter)
        filename = out_dir + 'epoch_{}.pth'.format(self.c_epoch + 1)
        optimizer = self.optimizer
        save_checkpoint(filename, self.model, optimizer, meta)
        
    
    def resume_training(self, checkpoint_path, map_location='default'):
        # 先加载checkpoint文件
        if map_location == 'default':
            device_id = torch.cuda.current_device()  # 获取当前设备
            checkpoint = load_checkpoint(self.model, 
                                         checkpoint_path, 
                                         map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = load_checkpoint(self.model,
                                         checkpoint_path, 
                                         map_location=map_location)
        #再恢复训练数据
        self.c_epoch = checkpoint['meta']['c_epoch'] + 1  # 注意：保存的是上一次运行的epoch，所以恢复要从下一次开始
        self.c_iter = checkpoint['meta']['c_iter'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info('resumed epoch %d, iter %d', self.c_epoch, self.c_iter)
    
    
    def _load_checkpoint(self, checkpoint_path, map_location):
        self.logger.info('load checkpoint from %s'%checkpoint_path)
        return load_checkpoint(self.model, checkpoint_path, map_location)


class TFRunner(Runner):
    """用于支持tensorflow的模型训练"""
    pass
    

class MXRunner(Runner):
    """用于支持mxnet的模型训练"""
    pass
    

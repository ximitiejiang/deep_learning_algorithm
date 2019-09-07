#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:46:38 2019

@author: ubuntu
"""
import torch
import torch.nn as nn
import os

from utils.prepare_training import get_config, get_logger, get_dataset, get_dataloader 
from utils.prepare_training import get_model, get_optimizer, get_lr_processor, get_loss_fn
from utils.visualization import visualization
from utils.tools import accuracy
from utils.checkpoint import load_checkpoint, save_checkpoint
    

class BatchProcessor():    
    def __call__(self):
        raise NotImplementedError('BatchProcessor class is not callable.')


class BatchDetector(BatchProcessor):    
    def __call__(self, model, data, training):
        losses = model(**data)  # 调用nn.module的__call__()函数，等效于调用forward()函数
        loss = losses.mean()
        outputs = dict(loss=loss)
        return outputs


class BatchClassifier(BatchProcessor):
    def __call__(self, model, data, loss_fn, device, training):
        img, label = data
        # 输入img要修改为float()格式float32，否则跟weight不匹配报错
        # 输入label要修改为long()格式int64，否则跟交叉熵公式不匹配报错
        img = img.float().to(device)
        label = label.long().to(device)
        # 计算输出
        y_pred = model(img)
        # 计算损失(!!!注意，一定要得到标量loss)
        loss = loss_fn(y_pred, label)  # pytorch交叉熵包含了前端的softmax/one_hot以及后端的mean
        acc1 = accuracy(y_pred, label, topk=1)
        # 更新反向传播
        loss.backward()                  # 用数值loss进行backward()      
        outputs = dict(loss=loss, acc1=acc1)
        return outputs


def get_batch_processor(cfg):
    if cfg.task == 'classifier':
        return BatchClassifier()
    elif cfg.task == 'detector':
        return BatchDetector()
    else:
        raise ValueError('Wrong task input.')



class Runner():
    """创建一个runner类，用于服务pytorch中的模型训练和验证: 支持cpu/单gpu/多gpu并行训练/分布式训练
    Runner类用于操作一个主模型，该主模型可以是单个分类模型，也可以是一个集成检测模型。
    所操作的主模型需要继承自pytorch的nn.module，且包含forward函数进行前向计算。
    如果主模型是一个集成检测模型，则要求该主模型所包含的所有下级模型也要继承
    自nn.module且包含forward函数。
    """
    def __init__(self, cfg_path,):
        # 获得配置
        self.cfg = get_config(cfg_path)
        # 检查文件夹和文件是否合法
        self.check_dir_file(self.cfg)
        #设置设备
        if self.cfg.gpus > 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.logger.info('Training will start in GPU!')
        if self.cfg.gpus == 0:
            self.device = torch.device("cpu")
            self.logger.info('Training will start in CPU!')
        #设置logger
        self.logger = get_logger(self.cfg.logger)
        self.logger.info('start logging info.')
        #创建batch处理器
        self.batch_processor = get_batch_processor(self.cfg)
        #创建数据集
        trainset = get_dataset(self.cfg.trainset, self.cfg.transform)
#        testset = get_dataset(self.cfg.testset, self.cfg.transform)
        #创建数据加载器
        self.dataloader = get_dataloader(trainset, self.cfg.trainloader)
        # 创建模型并初始化
        self.model = get_model(self.cfg)
        # 创建损失函数
        self.loss_fn_clf = get_loss_fn(self.cfg.loss_clf)
        if self.cfg.get('loss_reg', None) is not None:
            self.loss_fn_reg = get_loss_fn(self.cfg.loss_reg)
        # 优化器：必须在model送入cuda之前创建
        self.optimizer = get_optimizer(self.cfg.optimizer, self.model)
        # 学习率调整器
        self.lr_processor = get_lr_processor(self, self.cfg.lr_processor)
        # 恢复训练：加载模型参数+训练参数
        if self.cfg.resume_from:
            self.resume_training()
        # 加载参数：只加载模型参数
        elif self.cfg.load_from:
            self.load_checkpoint()
        # 送入GPU
        # TODO: 包装并行模型是在optimizer提取参数之前还是之后？
        if self.cfg.parallel and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)        
        # 共享变量
        self.buffer = {'loss': [],
                       'acc': [],
                       'lr':[]}
    
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
        if cfg.trainset.params.get('root_path', None) is not None:
            dir = cfg.trainset.params.root_path
            if not os.path.isdir(dir):
                raise FileNotFoundError('trainset path is not a dir.')
        if cfg.testset.params.get('root_path', None) is not None:
            dir = cfg.testset.params.root_path
            if not os.path.isdir(dir):
                raise FileNotFoundError('testset path is not a dir.')
    
    def current_lr(self):
        """获取当前学习率: 其中optimizer.param_groups有可能包含多个groups(但在我的应用中只有一个group)
        也就是为一个单元素list, 取出就是一个dict.
        即param_groups[0].keys()就包括'params', 'lr', 'momentum','dampening','weight_decay'
        返回：
            current_lr(list): [value0, value1,..], 本模型基本都是1个value，除非optimizer初始化传入多组lr
        """
        return [group['lr'] for group in self.optimizer.param_groups]  # 取出每个group的lr返回list，大多数情况下，只有一个group
    
    def train(self):
        """用于模型在训练集上训练"""
        self.model.train() # module的通用方法，可自动把training标志位设置为True
        n_epoch = 0
        self.lr_processor.set_base_lr_group()  # 设置初始学习率(直接从optimizer读取到：所以save model时必须保存optimizer) 
        while n_epoch < self.cfg.n_epochs:
            self.current_epoch = n_epoch   # 0->n-1
            self.lr_processor.set_regular_lr_group()  # 设置常规学习率(计算出来并填入optimizer)
            for n_iter, data_batch in enumerate(self.dataloader):
                self.current_iter = n_iter # 0->n-1
                self.lr_processor.set_warmup_lr_group() # 设置热身学习率(计算出来并填入optimizer)
                # 前向计算
                outputs = self.batch_processor(self.model, data_batch, 
                                               self.loss_fn_clf,
                                               self.device, training=True)
                # 反向传播
                self.optimizer.step()   
                self.optimizer.zero_grad()       # 每个batch的梯度清零
                # 显示text
                if (n_iter+1)%self.cfg.logger.interval == 0:
                    lr_str = ','.join(['{:.4f}'.format(lr) for lr in self.current_lr()]) # 用逗号串联学习率得到一个字符串
                    log_str = 'Epoch [{}][{}/{}]\tloss: {:.4f}\tacc: {:.4f}\tlr: {}'.format(
                            n_epoch+1, n_iter+1, len(self.dataloader),
                            outputs['loss'], outputs['acc1'], lr_str)

                    self.logger.info(log_str)
                # 存放结果
                self.buffer['loss'].append(outputs['loss'])
                self.buffer['acc'].append(outputs['acc1'])
                self.buffer['lr'].append(self.current_lr()[0])
            n_epoch += 1
            # 绘图
        visualization(self.buffer, title='train')
    
    def save_training(self, out_dir):
        meta = dict(current_epoch = self.current_epoch,
                    current_iter = self.current_iter)
        filename = '{}'
        
        save_checkpoint(filename, self.model, self.optimizer, meta)
        
    
    def resume_training(self, checkpoint_path, map_location='default'):
        # 先加载checkpoint文件
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = load_checkpoint(self.model, 
                                         checkpoint_path, 
                                         map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = load_checkpoint(self.model,
                                         checkpoint, 
                                         map_location=map_location)
        #再恢复训练数据
        
        
            
    def evaluate(self):
        """用于模型在验证集上验证和评估mAP"""
        self.model.eval() #
        for i, data_batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                outputs = self.batch_processor(self.model, data_batch, 
                                               self.loss_fn_clf,
                                               self.device,
                                               training=False)
                self.buffer['loss'].append(outputs['loss'])
                self.buffer['acc'].append(outputs['acc1'])
        visualization(self.buffer, title='test')
                
        
    def predict_single(self):
        """用于模型
        """
        self.model.eval()
        for i, data in enumerate(self.dataloader):
            pass
    
    def predict(self):
        pass
    


class TFRunner(Runner):
    """用于支持tensorflow的模型训练"""
    pass
    

class MXRunner(Runner):
    """用于支持mxnet的模型训练"""
    pass
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:46:38 2019

@author: ubuntu
"""
import torch
import torch.nn.functional as F

from utils.prepare_training import get_config, get_logger, get_dataset, get_dataloader 
from utils.prepare_training import get_model, get_optimizer




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
    def __call__(self, model, data, training):
        img, label = data
        # 计算输出
        y_pred = model(img)
        loss = F.cross_entropy(y_pred, label)  # pytorch交叉熵包含了前端的softmax/one_hot以及后端的mean
        acc1 = accuracy(y_pred, label, topk=1)
        acc5 = accuracy(y_pred, label, topk=5)
        # 更新反向传播
        self.optimizer.zero_grad()       # 每个batch的梯度清零
        self.outputs['loss'].backward()  # 这句话的执行是怎么操作？
        self.optimizer.step()     
        
        outputs = dict(loss=loss, acc1=acc1, acc5=acc5)
        return outputs

def get_batch_processor(cfg):
    if cfg.task == 'classifier':
        return BatchClassifier()
    elif cfg.task == 'detector':
        return BatchDetector()
    else:
        raise ValueError('Wrong task input.')

def accuracy(y_pred, label, topk=1):
    """pytorch tensor版本的精度计算
    y_pred()
    label()
    """
    with torch.no_grad():
        result=[]
        
        
        return result

class Runner():
    """创建一个runner类，用于服务pytorch中的模型训练和验证。
    Runner类用于操作一个主模型，该主模型可以是单个分类模型，也可以是一个集成检测模型。
    所操作的主模型需要继承自pytorch的nn.module，且包含forward函数进行前向计算。
    如果主模型是一个集成检测模型，则要求该主模型所包含的所有下级模型也要继承
    自nn.module且包含forward函数。
    """
    def __init__(self, cfg_path,):
        # 获得配置
        cfg = get_config(cfg_path)
        # 设置后端
        
        #设置logger
        self.logger = get_logger(cfg.log_level)
        #batch处理
        self.batch_processor = get_batch_processor(cfg)
        #创建数据集
        dataset = get_dataset(cfg.dataset)
        self.dataloader = get_dataloader(dataset, cfg.dataloader)
        # 创建模型
        self.model = get_model(cfg)
        # 优化器：必须在model送入cuda之前创建
        self.optimizer = get_optimizer(cfg.optimizer, self.model)
        # 恢复训练：加载模型参数+训练参数
        if cfg.resume_from:
            self.resume_training()
        # 加载参数：只加载模型参数
        elif cfg.load_from:
            self.load_checkpoint()
        # 送入GPU
        if torch.cuda.is_available and cfg.gpus:
            self.model = self.model.cuda()
            self.logger.info('Training will start in GPU!')
        else:
            self.logger.info('Training will start in CPU!')
        
    def train(self):
        """用于模型在训练集上训练"""
        self.model.train() # module的通用方法，可自动把training标志位设置为True
        for i, data_batch in enumerate(self.dataloader):
            # 计算输出
            outputs = self.batch_processor(self.model, data_batch, training=True)
            
                
    def evaluate(self):
        """用于模型在验证集上验证和评估mAP"""
        self.model.eval() #
        for i, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                outputs = self.batch_process(self.model, data_batch, training=False)
            
        return outputs
        
    def predict_single(self):
        """用于模型
        """
        self.model.eval()
        for i, data in enumerate(self.dataloader):
            pass
    
    def preddict(self):
        pass
    


class TFRunner(Runner):
    """用于支持tensorflow的模型训练"""
    pass
    
class CFRunner(Runner):
    """用于支持caffe的模型训练"""
    pass

class MXRunner(Runner):
    """用于支持mxnet的模型训练"""
    pass
    

# %%                
if __name__ == "__main__":
    runner = Runner(cfg_path = 'cfg_classifier_alexnet8_mnist.py')
    runner.train()
    runner.evaluate()
    runner.predict()
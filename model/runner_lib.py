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

from utils.prepare_training import get_config, get_logger, get_dataset, get_dataloader 
from utils.prepare_training import get_model, get_optimizer, get_lr_processor, get_loss_fn
from utils.visualization import vis_loss_acc
from utils.tools import accuracy
from utils.checkpoint import load_checkpoint, save_checkpoint
    

class BatchProcessor(): 
    """batchProcessor独立出来，是为了让Runner更具有通用性"""
    def __call__(self):
        raise NotImplementedError('BatchProcessor class is not callable.')


class BatchDetector(BatchProcessor):    
    def __call__(self, model, data, device):
        # 数据送入设备：注意数据格式问题改为在to_tensor中结果
        imgs = data['img'].to(device)
        img_metas = data['img_meta'].to(device)
        gt_bboxes = data['gt_bboxes'].to(device)
        gt_labels = data['gt_labels'].to(device)
        
        losses = self.model.bbox_head(imgs, gt_labels, gt_bboxes, img_metas)  # 调用nn.module的__call__()函数，等效于调用forward()函数
        loss = losses.mean()
        outputs = dict(loss=loss)
        return outputs


class BatchClassifier(BatchProcessor):
    def __call__(self, model, data, loss_fn, device, return_loss=True):
        img = data['img']
        label = data['label']
        # 输入img要修改为float()格式float32，否则跟weight不匹配报错，这步放到transform中to_tensor完成去了
        # 输入label要修改为long()格式int64，否则跟交叉熵公式不匹配报错，这步放到transform中to_tensor完成去了
        
        # 由于model要送入device进行计算，且该计算只跟img相关，跟label无关，所以可以只送img到device，也就是说label可以不组合成一个变量。
        img = img.to(device)
#        label = label.to(device)
        # 前向计算
        # 注意：一个检测问题中，每张图就是一个多样本分类问题，处理方式类似于一次分类，
        # 所以这里需要先组合label，多张图多个label也就等效于检测中的一张图多个bbox对应多个label
        y_pred = model(img).cpu()                     # 剩余计算都在cpu上进行
        label = torch.cat(label, dim=0)               # label组合(b,)
        acc1 = accuracy(y_pred, label, topk=1)
        outputs = dict(acc1=acc1)
        # 反向传播
        if return_loss:
            # 计算损失(!!!注意，一定要得到标量loss)
            loss = loss_fn(y_pred, label)  # pytorch交叉熵包含了前端的softmax/one_hot以及后端的mean
            loss.backward()  # 更新反向传播, 用数值loss进行backward()      
            outputs.update(loss=loss)
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
        # 共享变量: 需要声明在resume/load之前，否则会把resume的东西覆盖
        self.c_epoch = 0
        self.c_iter = 0
        self.weight_ready = False
        self.buffer = {'loss': [],
                       'acc': [],
                       'lr':[]}
        # 获得配置
        self.cfg = get_config(cfg_path)
        # 检查文件夹和文件是否合法
        self.check_dir_file(self.cfg)
        #设置logger
        self.logger = get_logger(self.cfg.logger)
        self.logger.info('start logging info.')
        #设置设备
        if self.cfg.gpus > 0 and torch.cuda.is_available():
            self.device = torch.device("cuda")   # 设置设备GPU: "cuda"和"cuda:0"的区别？
            self.logger.info('Operation will start in GPU!')
        if self.cfg.gpus == 0:
            self.device = torch.device("cpu")      # 设置设备CPU
            self.logger.info('Operation will start in CPU!')
        #创建batch处理器
        self.batch_processor = get_batch_processor(self.cfg)
        #创建数据集
        self.trainset = get_dataset(self.cfg.trainset, self.cfg.transform)
        self.valset = get_dataset(self.cfg.valset, self.cfg.transform_val) # 做验证的变换只做基础变换，不做数据增强
        
#        data = self.trainset[0]
        
        #创建数据加载器
        self.dataloader = get_dataloader(self.trainset, self.cfg.trainloader)
        self.valloader = get_dataloader(self.valset, self.cfg.valloader)
        
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
        # 送入GPU
        # 包装并行模型是在optimizer提取参数之后，否则可能导致无法提取，因为并行模型在model之下加了一层module壳
        if self.cfg.parallel and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        # 注意：恢复或加载是直接加载到目标设备，所以必须在模型传入设备之后进行，确保设备匹配
        # 加载参数，从之前断开的位置继续训练
        if self.cfg.resume_from:
            self.resume_training(checkpoint_path=self.cfg.resume_from, 
                                 map_location=self.device)  # 沿用设置的device
        # 加载参数，一般用来做预测
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
    
    def train(self):
        """用于模型在训练集上训练"""
        self.model.train() # module的通用方法，可自动把training标志位设置为True
        self.lr_processor.set_base_lr_group()  # 设置初始学习率(直接从optimizer读取到：所以save model时必须保存optimizer) 
        start = time.time()
        while self.c_epoch < self.cfg.n_epochs:
            self.lr_processor.set_regular_lr_group()  # 设置常规学习率(计算出来并填入optimizer)
            for self.c_iter, data_batch in enumerate(self.dataloader):
                self.lr_processor.set_warmup_lr_group() # 设置热身学习率(计算出来并填入optimizer)
                # 前向计算
                outputs = self.batch_processor(self.model, data_batch, 
                                               self.loss_fn_clf,
                                               self.device,
                                               return_loss=True)
                # 反向传播
                self.optimizer.step()   
                self.optimizer.zero_grad()       # 每个batch的梯度清零
                # 显示text
                if (self.c_iter+1)%self.cfg.logger.interval == 0:
                    lr_str = ','.join(['{:.4f}'.format(lr) for lr in self.current_lr()]) # 用逗号串联学习率得到一个字符串
                    log_str = 'Epoch [{}][{}/{}]\tloss: {:.4f}\tacc: {:.4f}\tlr: {}'.format(
                            self.c_epoch+1, self.c_iter+1, len(self.dataloader),
                            outputs['loss'], outputs['acc1'], lr_str)

                    self.logger.info(log_str)
                # 存放结果
                self.buffer['loss'].append(outputs['loss'])
                self.buffer['acc'].append(outputs['acc1'])
                self.buffer['lr'].append(self.current_lr()[0])
            # 保存模型
            if self.c_epoch%self.cfg.save_checkpoint_interval == 0:
                self.save_training(self.cfg.work_dir)            
            self.c_epoch += 1
        times = time.time() - start
        self.logger.info('training finished with times(s): {}'.format(times))
        # 绘图
        vis_loss_acc(self.buffer, title='train')
        self.weight_ready= True
    
    
    def evaluate(self):
        """针对数据集的预测：用于模型在验证集上验证和acc评估: 
        注意：需要训练完成后，或在cfg中设置load_from，也就是model先加载训练好的参数文件。
        """
        self.buffer = {'acc': []}  # 重新初始化buffer
        self.n_correct = 0    # 用于计算全局acc
        if self.weight_ready:
            self.model.eval()   # 关闭对batchnorm/dropout的影响，不再需要手动传入training标志
            for c_iter, data_batch in enumerate(self.valloader):
                with torch.no_grad():  # 停止反向传播，只进行前向计算
                    outputs = self.batch_processor(self.model, data_batch, 
                                                   self.loss_fn_clf,
                                                   self.device,
                                                   return_loss=False)
                    self.buffer['acc'].append(outputs['acc1'])
                # 计算总体精度
                self.n_correct += self.buffer['acc'][-1] * len(data_batch[0])
            
            vis_loss_acc(self.buffer, title='val')
            self.logger.info('ACC on valset: %.3f', self.n_correct/len(self.valset))
        else:
            raise ValueError('no model weights loaded.')
    
    def predict_single(self, img):
        """针对单个样本的预测：也是最精简的一个预测流程，因为不创建数据集，不进入batch_processor.
        直接通过model得到结果，且支持cpu/GPU预测。
        注意：需要训练完成后，或在cfg中设置load_from，也就是model先加载训练好的参数文件。
        """
        from utils.transformer import ImgTransform
        if self.weight_ready:
            img_transform = ImgTransform(self.cfg.transform_val)
            img, *_ = img_transform(img)
            img = img.float().to(self.device)
            # 前向计算
            y_pred = self.model(img)
            y_class = self.trainset.CLASSES[y_pred]
            self.logger.info('predict class: %s', y_class)
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
    

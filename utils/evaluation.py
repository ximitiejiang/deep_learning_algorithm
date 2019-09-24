#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:15:27 2019

@author: ubuntu
"""
import torch
from utils.prepare_training import get_config, get_dataset, get_dataloader, get_model
from utils.checkpoint import load_checkpoint
from utils.transform import to_device
from utils.visualization import vis_loss_acc
from utils.tools import accuracy

# %% 评估数据集
def eval_dataset_cls(cfg_path, device=None):
    """分类问题的eval dataset: 
    等效于runner中的load_from + val，但可用来脱离runner进行独立的数据集验证
    """
    # 准备验证所用的对象
    cfg = get_config(cfg_path)
    dataset = get_dataset(cfg.valset, cfg.transform_val)
    dataloader = get_dataloader(dataset, cfg.valloader)
    model = get_model(cfg)
    if device is None:
        device = torch.device(cfg.load_device)
    # TODO: 如下两句的顺序
    load_checkpoint(model, cfg.load_from, device)
    model = model.to(device)
    # 开始验证
    buffer = {'acc': []}
    n_correct = 0
    model.eval()
    for c_iter, data_batch in enumerate(dataloader):
        with torch.no_grad():  # 停止反向传播，只进行前向计算
            img = to_device(data_batch['img'], device)
            label = to_device(data_batch['gt_labels'], device)
            
            y_pred = model(img)
            label = torch.cat(label, dim=0)
            acc1 = accuracy(y_pred, label, topk=1)
            buffer['acc'].append(acc1)
        # 计算总体精度
        n_correct += buffer['acc'][-1] * len(data_batch['gt_labels'])
    
    vis_loss_acc(buffer, title='eval dataset')
    print('ACC on dataset: %.3f', n_correct/len(dataset))


def eval_dataset_det(cfg_path, device=None):
    """检测问题的eval dataset
    等效于runner中的load_from + val + bbox
    """
    # 准备验证所用的对象
    cfg = get_config(cfg_path)
    dataset = get_dataset(cfg.valset, cfg.transform_val)
    dataloader = get_dataloader(dataset, cfg.valloader)
    model = get_model(cfg)
    if device is None:
        device = torch.device(cfg.load_device)
    # TODO: 如下两句的顺序
    load_checkpoint(model, cfg.load_from, device)
    model = model.to(device)
    # 开始验证
    buffer = {'acc': []}
    n_correct = 0
    model.eval()
    for c_iter, data_batch in enumerate(dataloader):
        with torch.no_grad():  # 停止反向传播，只进行前向计算
            img = to_device(data_batch['img'], device)
            label = to_device(data_batch['gt_labels'], device)

            y_pred = model(img)
            label = torch.cat(label, dim=0)
            acc1 = accuracy(y_pred, label, topk=1)
            buffer['acc'].append(acc1)
        # 计算总体精度
        n_correct += buffer['acc'][-1] * len(data_batch['gt_labels'])
    
    
    print('ACC on dataset: %.3f', n_correct/len(dataset))


# %% 预测单张图片
def predict_one_img_cls(img):
    """针对单个样本的预测：也是最精简的一个预测流程，因为不创建数据集，不进入batch_processor.
    直接通过model得到结果，且支持cpu/GPU预测。
    注意：需要训练完成后，或在cfg中设置load_from，也就是model先加载训练好的参数文件。
    """
    from utils.transformer import ImgTransform

    img_transform = ImgTransform(cfg.transform_val)
    img, *_ = img_transform(img)
    img = img.to(device)
    # 前向计算
    y_pred = model(img)
    y_class = trainset.CLASSES[y_pred]
    print('predict class: %s', y_class)

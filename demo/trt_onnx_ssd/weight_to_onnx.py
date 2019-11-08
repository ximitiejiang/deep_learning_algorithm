#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:55:04 2019

@author: ubuntu
"""
import torch
from utils.prepare_training import get_config, get_model
from utils.checkpoint import load_checkpoint


"""
把一个pytorch模型转换成onnx模型：
"""
# 创建配置和创建模型
cfg_path = './cfg_detector_ssdvgg16_voc.py'
cfg = get_config(cfg_path)
cfg.load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_61.pth'
model = get_model(cfg)
# 加载权重
load_checkpoint(model, cfg.load_from)

# 生成验证数据
imgs = torch.randn(1, 3, 300, 300).cuda()
#img_metas
#gt_bboxes
#gt_labels
dummy_input = imgs
model = model.cuda()
# 导出onnx和验证数据
onnx_path = '/home/ubuntu/mytrain/onnx_ssd/ssd.onnx'
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

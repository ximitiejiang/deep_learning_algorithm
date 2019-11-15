#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:55:04 2019

@author: ubuntu
"""
import torch
from utils.prepare_training import get_config, get_model
from utils.checkpoint import load_checkpoint


def onnx_exporter(cfg, output_path):
    """把一个pytorch模型转换成onnx模型，对模型的要求：
    1. 模型需要有forward_dummy()函数的实施，如下是一个实例：
    def forward_dummy(self, img):
        x = self.extract_feat(img)
        x = self.bbox_head(x)
        return x
    2. 模型的终端输出，也就是head端的输出必须是tuple/list/variable类型，不能是dict，否则当前pytorch.onnx不支持。
    """
    img_shape = (1, 3) + cfg.img_size
    dummy_input = torch.randn(img_shape, device='cuda')
    
    # 创建配置和创建模型
    model = get_model(cfg).cuda()
    if cfg.load_from is not None:
        _ = load_checkpoint(model, cfg.load_from)
    else:
        raise ValueError('need to assign checkpoint path to load from.')
    
    model.forward = model.forward_dummy
    torch.onnx.export(model, dummy_input, output_path, verbose=True)
    

if __name__ == "__main__":
    cfg_path = './cfg_detector_ssdvgg16_voc.py'
    cfg = get_config(cfg_path)
    cfg.load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_61.pth'
    output_path = '/home/ubuntu/mytrain/onnx_ssd/ssd.onnx'
    onnx_exporter(cfg, output_path)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:22:46 2019

@author: ubuntu
"""

from utils.module_factory import registry, build_module
import torch.nn as nn

# %% onestage
class OneStageDetector(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 创建基础模型
        self.backbone = build_module(cfg.model.backbone, registry)
        self.bbox_head = build_module(cfg.model.bbox_head, registry)
        if cfg.model.neck is not None:
            self.neck = build_module(cfg.model.neck, registry)
        
        # 初始化
        # TODO: init weight中我没有指定map_location，那么指定cpu/gpu操作需要在之后进行
        self.init_weights(pretrained=cfg.model.pretrained)
    
    def init_weights(self, pretrained):
        self.backbone.init_weights(pretrained = pretrained)
        self.bbox_head.init_weights()
        if self.neck is not None:
            self.neck.init_weights()  # TODO: 检查为什么上一版本没有这句neck的初始化
        
        
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """nn.Module的forward()函数，分支成训练的forward()以及测试的forward()"""
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, rescale=False, **kwargs)  # TODO: 是否可以去掉rescale
        
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels):
        """训练过程的前向计算的底层函数"""
        # 特征提取
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        outs = self.bbox_head(x)
        # 计算损失
        bbox_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.cfg.train_cfg)
        losses = self.bbox_head.get_losses(*bbox_inputs)
        return losses
        
    def forward_test(self, img, img_metas, rescale=False, **kwargs):
        """测试过程的前向计算的底层函数: 只支持单张图片，如果多张图片则需要自定义循环"""
        # TODO: 测试过程只需要前向计算而不需要反向传播，是否可以缩减模型尺寸?
        # 特征提取
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        outs = self.bbox_head(x)
        # 计算bbox
        bbox_inputs = outs + (img_metas, self.cfg.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(**bbox_inputs)        
        # TODO: bbox形式转换还需要增加
        return bbox_list
    
    def show_result(self):
        pass
    

# %% two stage
class TwoStageDetector(nn.Module):
    def __init__(self):
        super().__init__()
        pass


# %%
class Classifier(nn.Module):
    """用于分类器的总成模型"""
    def __init__(self):
        super().__init__()
        
    
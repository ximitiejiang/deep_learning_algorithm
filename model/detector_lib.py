#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:22:46 2019

@author: ubuntu
"""

#from utils.module_factory import registry, build_module
from utils.prepare_training import get_model
import torch.nn as nn

# %% onestage
class OneStageDetector(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 创建基础模型
        self.backbone = get_model(cfg.backbone)
        if cfg.neck is not None:
            self.neck = get_model(cfg.neck)
        self.bbox_head = get_model(cfg.head)
        
        # 初始化: 注意权重需要送入cpu/gpu，该步在model.to()完成
        self.init_weights()
    
    def init_weights(self):
        self.backbone.init_weights(pretrained = self.cfg.backbone.params.pretrained)
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
        
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:22:46 2019

@author: ubuntu
"""

#from utils.module_factory import registry, build_module

import torch.nn as nn

    
# %% onestage
class OneStageDetector(nn.Module):
    """生成单阶段的检测器：
    单阶段检测器组成：backbone + (neck) + bbox_head, 带bbox_head说明输出的就是bbox预测了
    双阶段检测器组成：backbone + (neck) + anchor_head + bbox_head, 带anchor_head说明输出的就是anchor proposal
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 创建基础模型
        from utils.prepare_training import get_model
        self.backbone = get_model(cfg.backbone)
        if cfg.neck:  # 不能写成 is not None, 因为结果是{}不是None, 但可以用True/False来判断
            self.neck = get_model(cfg.neck)
        self.bbox_head = get_model(cfg.head)  # bbox head就说明生成的是bbox
        
        # 初始化: 注意权重需要送入cpu/gpu，该步在model.to()完成
        self.init_weights()
    
    def init_weights(self):
        self.backbone.init_weights()
        if self.cfg.neck:
            self.neck.init_weights()
        self.bbox_head.init_weights()
        
        
    def forward(self, imgs, img_metas, return_loss=True, **kwargs):
        """nn.Module的forward()函数，分支成训练的forward()以及测试的forward()"""
        if return_loss:
            return self.forward_train(imgs, img_metas, **kwargs)
        else:
            return self.forward_test(imgs, img_metas, rescale=False, **kwargs)  # TODO: 是否可以去掉rescale
        
    def forward_train(self, imgs, img_metas, gt_bboxes, gt_labels):
        """训练过程的前向计算的底层函数"""
        # 特征提取
        x = self.backbone(imgs)
        if self.cfg.neck:
            x = self.neck(x)
        outs = self.bbox_head(x)  # 获得分类和回归的预测值cls_score, bbox_preds
        # 计算损失
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.cfg)
        losses = self.bbox_head.get_losses(*loss_inputs)
        return losses
        
    def forward_test(self, imgs, img_metas, rescale=False, **kwargs):
        """测试过程的前向计算的底层函数: 只支持单张图片，如果多张图片则需要自定义循环
        """
        # TODO: 测试过程只需要前向计算而不需要反向传播，是否可以缩减模型尺寸?
        # 特征提取
        x = self.backbone(imgs)
        if self.cfg.neck:
            x = self.neck(x)
        outs = self.bbox_head(x)
        # 计算bbox
        bbox_inputs = outs + (img_metas, self.cfg)
        bboxes, labels = self.bbox_head.get_bboxes(*bbox_inputs)        
        return bboxes, labels
    
    

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
        
    
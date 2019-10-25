#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:34:58 2019

@author: ubuntu
"""
import torch
import torch.nn as nn
from model.loss_lib import SigmoidFocalLoss, IouLoss, SigmoidBinaryCrossEntropyLoss
from utils.init_weights import normal_init, bias_init_with_prob
from model.get_target_lib import get_points, get_point_target, get_centerness_target
from model.bbox_regression_lib import lrtb2bbox


def conv3x3_group(inc, outc, stride=1, padding=1):
    """带组归一化的卷积3x3"""
    return nn.Sequential(
            nn.Conv2d(inc, outc, 3, stride=stride, padding=padding),
            nn.GroupNorm(32, outc),
            nn.ReLU(inplace=True))
    

class ClassHead(nn.Module):
    """分类模块: 负责单层特征层"""
    def __init__(self, in_channels, num_classes=21, stacked_convs=4):
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs = nn.ModuleList()
        for _ in range(stacked_convs):
            self.cls_convs.append(conv3x3_group(in_channels, in_channels, 1, 1))

    def forward(self, x):
        for cls_conv in self.cls_convs:
            x = cls_conv(x)                   # (b,256,w,h)
        return x


class BboxHead(nn.Module):
    """bbox回归模块: 负责单层特征"""
    def __init__(self, in_channels, stacked_convs=4):
        super().__init__()
        self.bbox_convs = nn.ModuleList()
        for _ in range(stacked_convs):
            self.bbox_convs.append(conv3x3_group(in_channels, in_channels, 1, 1))

    def forward(self, x):
        for bbox_conv in self.bbox_convs:
            x = bbox_conv(x)                         # (b,256,h,w)
        return x      


class FCOSHead(nn.Module):
    """fcos无anchor的head
    """
    def __init__(self,
                 num_classes=21,
                 in_channels=256,  # 堆叠卷积的输入通道数(不包括做变换的卷积层)
                 out_channels=256, # 堆叠卷积的输出通道数
                 stacked_convs=4,      # 堆叠的卷积层数
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8))
                 ):
        
        super().__init__()
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.regress_ranges = regress_ranges
        self.strides = strides
        # 创建分类回归层：每层特征独享一个子头
        num_featmaps = len(strides)
        self.cls_heads = nn.ModuleList()
        self.bbox_heads = nn.ModuleList()
        for _ in range(num_featmaps):
            self.cls_heads.append(ClassHead(in_channels, num_classes, stacked_convs))
            self.bbox_heads.append(BboxHead(in_channels, stacked_convs))
        # 调整层：注意调整层是共享的(独享一般也可以，多一点点参数而已)
        self.fcos_cls = nn.Conv2d(in_channels, num_classes - 1, 3, stride=1, padding=1)
        self.fcos_centerness = nn.Conv2d(in_channels, 1, 3, stride=1, padding=1)
        self.fcos_reg = nn.Conv2d(in_channels, 4, 3, stride=1, padding=1)
        # 这里没有采用scales层
        
        # 创建损失函数
        self.loss_cls = SigmoidFocalLoss()
        self.loss_reg = IouLoss()
        self.loss_centerness = SigmoidBinaryCrossEntropyLoss()


    def init_weights(self):
        for m in self.cls_heads.modules():
            normal_init(m.conv, std=0.01)
        for m in self.bbox_heads.modules():
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_heads.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)
    
    
    def forward(self, x):
        """对FPN过来的输入进行计算
        args:
            x: 表示5层特征(5,)，每层(b, 256, h, w)
        """
        # 分别计算每一张特征图(b,c,h,w), 共计5张
        scores = [cls_head(feat) for cls_head, feat in zip(self.cls_heads, x)]
        cls_scores = [self.fcos_cls(score) for score in scores]
        centerness = [self.fcos_centerness(score) for score in scores]
        
        bbox_preds = [bbox_head(feat) for bbox_head, feat in zip(self.bbox_heads, x)]
        bbox_preds = [self.fcos_reg(bbox_pred).float().exp() for bbox_pred in bbox_preds]
        
        # (PRC)permute + reshape + concate
        cls_scores = [cls_score.permute(0,2,3,1).reshape(-1, self.num_classes -1) for cls_score in cls_scores]  #(5,)(b,-1,20)
        centerness = [center.permute(0,2,3,1).reshape(-1) for center in centerness]                             #(5,)(b,-1)
        bbox_preds = [bbox_pred.permute(0,2,3,1).reshape(-1, 4) for bbox_pred in bbox_preds]                    #(5,)(b,-1,4)
        
        # 堆叠
        cls_scores = torch.cat(cls_scores)  # (b, -1, 20)
        centerness = torch.cat(centerness)  # (b,-1)
        bbox_preds = torch.cat(bbox_preds)  # (b, -1, 4)
        return cls_scores, bbox_preds, centerness 
    
    
    def get_losses(self, cls_scores, bbox_preds, centernesses, 
                   gt_bboxes, gt_labels, img_metas, cfg):
        """计算损失：先获得target，然后基于样本和target计算损失
        注意：对FCOS的loss可以一次性把一个batch的loss一起算出来，这是更高效的算法
        args:
            cls_scores: (5,)(b, 20, h, w) 按层分组
            bbox_preds: (5,)(b, 4, h, w) 按层分组
            centernesses: (5,)(b, 1, h, w) 按层分组
            gt_bboxes: (b,)(k, 4) 按batch分组
            gt_labels: (b,)(k, ) 按batch分组
            img_metas: (b,) 按batch分组
        """
        num_imgs = len(img_metas)
        featmap_sizes = [featmap.shape[-2:] for featmap in cls_scores] # (5,) (h,w)
        device = cls_scores.get_device()
        # 计算网格中心点
        points = get_points(featmap_sizes, self.strides, device) # (5,)(k,2)
        num_level_points = [pt.shape[0] for pt in points]

        # 计算target
        labels, bbox_targets = get_point_target(
                points, self.regress_ranges, gt_bboxes, num_level_points) # labels(5,)(m,),  bbox_targets(5,)(m,4) 
        
        # 展平调整格式: 统一为外层为特征层数，内层为(b, xxx), 然后堆叠
        # 注意：这个展平方式有点特殊，常规的格式调整是从特征层索引(5,)变换到batch索引(b,)，而这里省略这一步，而是直接跳到下一步，再把batch索引也合并到个数中。
        all_cls_scores = torch.cat([
                cls_score.permute(0,2,3,1).reshape(-1, self.num_classes - 1)   # 注意，分类score里边是按照20类或80类来算，不考虑负样本。因为采用的损失函数是focal loss而不是交叉熵。
                for cls_score in cls_scores])             # (5,)(b,20,h,w)->(5*b*h*w,20)
        all_bbox_preds = torch.cat([
                bbox_pred.permute(0,2,3,1).reshape(-1, 4)  
                for bbox_pred in bbox_preds])             # (5,)(b,4,h,w)->(5*b*h*w,4)
        all_centernesses = torch.cat([
                centerness.permute(0,2,3,1).reshape(-1)
                for centerness in centernesses])          # (5,)(b,1,h,w)->(5*b*h*w,20)
        # 注意points的堆叠，需要在内层先配出batch的数量出来,然后再按照外层的特征层数堆叠
        all_points = torch.cat([
                point.repeat(num_imgs, 1) for point in points]) # (5,)(k,2) -> (5*k, 2)
        all_labels = torch.cat(labels)    # (5*k, )
        all_bbox_targets = torch.cat(bbox_targets) # (5*k, 4)
        
        # 分类损失: 采用focal loss
        pos_inds = all_labels.nonzero().reshape(-1)  # 得到正样本的index
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(all_cls_scores,                # ()
                                 all_labels,                    # () 
                                 avg_factor=num_pos + num_imgs)  # 增加num_imgs是防止分母为0 

        # 计算中心点分类损失
        pos_bbox_preds = all_bbox_preds[pos_inds]      # 得到正样本对应bbox_pred
        pos_bbox_targets = all_bbox_targets[pos_inds]  # 得到正样本对应bbox_target
        
        pos_centerness = all_centernesses[pos_inds]    # 得到正样本对应中心点
        pos_centerness_targets = get_centerness_target(pos_bbox_targets)
        
        if num_pos > 0: # 如果有正样本，则求位置损失
            pos_points = all_points[pos_inds]
            pos_decoded_bbox_preds = lrtb2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = lrtb2bbox(pos_points, pos_bbox_targets)
            # 回归损失采用-log(iou)作为损失函数，并且加入centerness作为权重
            loss_reg = self.loss_reg(pos_decoded_bbox_preds,   # (k, 4)
                                     pos_decoded_target_preds, # (k, 4)
                                     weight=pos_centerness_targets,  # (k,)
                                     avg_factor=pos_centerness_targets.sum())
            # 中心点损失采用二值交叉熵做损失函数：中心度最高是1， 相当于一个概率
            # 也就是评价预测的中心点的中心度与实际中心点的中心度之间的相关性。
            # 此时label不再是[0,1]这种二分类标签，而是[0~1]之间的一个值，所以属于不太典型的二值交叉熵计算
            # 此时应该相当于一个回归问题，该点的中心度是一个回归值，使用smooth_l1也许也可以。
            loss_centerness = self.loss_centerness(pos_centerness,         # (k,)
                                                   pos_centerness_targets) # (k,)
        else:
            loss_reg = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
        # 计算    
        return dict(loss_cls = loss_cls,
                    loss_reg = loss_reg,
                    loss_centerness = loss_centerness)
        
        def get_bboxes(self, cls_scores, bbox_preds, centernesses, img_metas, cfg):
            """
            Args:
                cls_scores(6,)(b,c,h,w): 按特征图分组
                bbox_preds(6,)(b,c,h,w):按特征图分组
            """
            pass
        
        def get_one_img_bboxes(self, cls_scores, bbox_preds, centernesses, 
                               points, img_shape, scale_factor, cfg):
            """
            Args:
                cls_scores(6,)(c,h,w): 按特征图分组
                bbox_preds(6,)(c,h,w):按特征图分组
            """
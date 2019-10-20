#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:15:42 2019

@author: ubuntu
"""
from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from model.get_target_lib import get_anchor_target
from model.anchor_generator_lib import AnchorGenerator
from utils.init_weights import xavier_init
from model.loss_lib import CrossEntropyLoss, SmoothL1Loss
from model.bbox_regression_lib import delta2bbox
from model.nms_lib import nms_operation
from model.bbox_regression_lib import lrtb2bbox
from model.loss_lib import IouLoss, SigmoidBinaryCrossEntropyLoss, SigmoidFocalLoss


"""待整理：
小物体检测效果不好的原因;
1. 模型学习到了，但没有检测到，这属于漏检：这时可能是anchor的设置不合理，通过调整anchor尺寸去改善。
2. 模型没有学习到，则检测出来的是错的或者置信度不高，这属于误检：这时可能是模型输出特征的层数太浅，导致没有学习到相应小物体，通过增加层数取改善。

"""


#def conv_head(in_channel_list, out_channel_list):
#    """用来创建并列的一组卷积分别对应并列的一组特征图(一一对应)，用来分类或者回归的通道数变换：
#    比如从通道数512变换成分类需要的通道数n_class*n_anchor, 或者变换成回归需要的通道数
#    """
#    n_convs = len(in_channel_list)
#    layers = []
#    for i in range(n_convs):
#        # 卷积参数是常规不改变w,h的参数(即s=1,p=1)
#        layers.append(nn.Conv2d(in_channel_list[i], out_channel_list[i], 
#                                kernel_size=3, padding=1, stride=1))
#    return layers
    

def get_base_anchor_params(img_size, ratio_range, n_featmap, strides, ratios):
    """这是SSD使用的一种通用的设计anchor尺寸的方法
    假定已知图像尺寸img_size：
    1.定义一个anchor相对于图像的尺寸比例范围ratio_range, 也就是最小anchor大概是图像的百分之多少，最大anchor大概是图像的百分之多少.
      这个比例通常是通过对数据集进行聚类来获得anchor相对原图的最小最大比例范围.
      但注意：直接用聚类结果设置anchor往往对小物体检测效果不好，因为小物体数量少可能被聚类忽略。
    2. 添加额外的针对小物体的比例进比例列表
    3. 
    Args:
        img_size
        ratio_range(list): (min_ratio, max_ratio)代表anchor是img的最小百分比，以及是img的最大百分比，比如(0.2, 0.9)
        n_featmap: 代表有多少张特征图，显然每张特征图对应的anchor基础尺寸不同
        strides: 代表每张特征图的缩放比例，也代表特征图上一个像素点代表原图上多少个像素点(即该特征图的感受野)，也代表
        ratios: 代表每张特征图上anchor在base_anchor基础上的比例变化
    """
    if not isinstance(img_size, (int, float)): # 如果输入二维size，只取一维，也就是只适合方形图片
        img_size = img_size[0]
    # 可以通过聚类来决定最小anchor和最大anchor的范围，但这会丢失一部分尺寸很小数量不多的小物体，
    # 所以把聚类得到的比例放在后边5层，并在第一层再手动insert一组小比例ratio
    min_ratio, max_ratio = ratio_range
    step = (max_ratio - min_ratio) / (n_featmap - 2)  # 基于特征图个数，定义一个阶梯，先保证后边5个特征图每个都有一个min_ratio
    min_ratios = [min_ratio + step * i for i in range(n_featmap - 1)]    # 定义后5个特征图min_ratio   
    max_ratios = [min_ratio + step * (i + 1) for i in range(n_featmap - 1)]  # 定义后5个特征图max_ratio
    # 为了提高小物体检测能力，对少量小物体再增加一组更小比例的anchor
    min_ratios.insert(0, 0.1)   # voc尺寸在300时，添加小尺寸anchor范围是0.1-0.2，voc尺寸512是图像加大，小尺寸anchor范围是0.07-0.15 
    max_ratios.insert(0, 0.2)
    # ratio转换成size
    min_sizes = [int(min_ratios[i] * img_size) for i in range(len(min_ratios))]
    max_sizes = [int(max_ratios[i] * img_size) for i in range(len(max_ratios))]
    # 生成anchor ratios
    anchor_ratios = []
    for rr in ratios:  # [2,3]
        anchor_ratio = [1.]
        for r in rr:
            anchor_ratio += [1/r, r]  # ratio范围定义[1, 1/2, 2, 1/3, 3]
        anchor_ratios.append(anchor_ratio)
    # 生成输出参数
    base_sizes = min_sizes   # 定义anchor基础尺寸
    anchor_scales = [(1., np.sqrt(max_sizes[i]/min_sizes[i])) for i in range(len(max_sizes))]  # 定义anchor缩放比例: 开根号是因为
    centers = [((strides[i] - 1)/2., (strides[i] - 1)/2.) for i in range(len(strides))]   # 定义anchor中心点坐标
    return (base_sizes,     # (30, 60, 112, 165, 217, 270)
            anchor_scales,  # ((1, 1.41),(1, 1.49),(1,1.2), (1, 1.16), (1, 1.12),(1, 1.09))
            anchor_ratios,  # (1, 1/2, 2, 1/3, 3)
            centers)        # ((3.5,3.5),(7.5,7.5),(15.5,15.5),(31.5,31.5),(49.5,49.5),(149.5,149.5))

def ohem(loss_cls, label_targets, neg_pos_ratio, avg_factor):
    """负样本挖掘：从中挖掘出分类损失中固定比例的，难样本的损失值作为负样本损失
    既保证正负样本平衡，也保证对损失贡献大的负样本被使用
    """    
    pos_inds = torch.nonzero(label_targets > 0).reshape(-1)
    neg_inds = torch.nonzero(label_targets == 0).reshape(-1)
    num_pos_samples = pos_inds.shape[0]
    num_neg_samples = neg_pos_ratio * num_pos_samples       # 计算需要的负样本个数
    if num_neg_samples > neg_inds.shape[0]:  # 如果需要的负样本数超过现有负样本数，则负样本就取现有负样本数(一般不会出现这种情况)
        num_neg_samples = neg_inds.shape[0]
    # 提取负样本，然后排序，然后抽样
    topk_loss_cls_neg, _ = loss_cls[neg_inds].topk(num_neg_samples) # 找到负样本对应loss，从中提取需要的负样本个数的loss
    loss_cls_pos = loss_cls[pos_inds]
    loss_cls_neg = topk_loss_cls_neg
    
    # loss均值缩减
    loss_cls_pos = loss_cls_pos.sum()
    loss_cls_neg = loss_cls_neg.sum()
    loss_cls = (loss_cls_pos + loss_cls_neg) / avg_factor
    
    return loss_cls
  
    
# %%     

class ClassHead(nn.Module):
    """针对单层特征的分类模块"""
    def __init__(self, in_channels, num_anchors, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv3x3 = nn.Conv2d(in_channels, num_anchors * num_classes, 3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv3x3(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, self.num_classes)
        return out
        

class BboxHead(nn.Module):
    """针对单层特征的bbox回归模块"""
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, num_anchors * 4, 3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv3x3(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, 4)
        return out     


class SSDHead(nn.Module):
    """分类回归头：
    分类回归头的工作过程： 以下描述在维度上都省略b，因为b在整个模型过程不变，只讨论单张图的情况
    step1. 从backbone获取多层特征图(512,38,38),(1024,19,19),(512,10,10),(256,5,5),(256,3,3),(256,1,1)
    
    step2. 采用卷积调整特征的层数到预测需要的形式(一个特征图对应一个卷积层即可)
        > 其中作为分类问题需要模型这个万能函数输出y_pred(n_anchors, n_classes)，
          n_anchors是总的anchor个数(8732个)，n_classes是类别数(21)类，这样才能跟labels(n_anchors)进行分类损失计算
          所以调整层数的逻辑就是原来的512层变成21*4层，这样结合特征尺寸(w,h)，就可以通过reshape凑出anchor个数4*w*h和类别数21.
        > 其中作为回归问题需要模型这个万能函数输出y_pred(n_anchors, n_coords)
          n_anchors是总的anchor个数(8732个)，n_coords是每个bbox坐标数(4个)，这样才能跟labels(n_anchors, n_coords)进行回归损失计算
          所以调整层数的逻辑就是原来的512层变成4*4层，这样结合特征尺寸(w,h)，就可以通过reshape凑出anchor个数4*w*h和坐标数4.
    
    step3. 采用anchor机制确定每个anchor的分类标签和回归标签: anchor机制是整个物体检测的核心，
        思想就是特征层上任何一个像素都在原图有一个对应感受野(8x8, 16x16, 32x32, 64x64, 128x128, 300x300)，对原图上每一个对应感受野上布置一组anchor，
        从而让anchor遍布整张原图，而每个anchor的大小都是根据数据集bbox尺寸聚类后设计出来的尺寸，跟感受野尺寸无关，但会把感受野中心作为anchor布置的中心。
        > 先生成base_anchor，然后扩展到grid_anchor(8732个)
        > 把所有anchors跟gt_bbox进行iou计算，评价出每个anchor的身份：iou>0.5的是正样本，其他是负样本。
        > 让正样本获得gt bbox的标签，负样本获得标签为0(这也是为什么分类要多一类变成21类或81类)，该标签就可以用来做分类的预测(计算acc，计算loss)
        > 让正样本获得gt bbox的坐标，负样本获得坐标为0，该坐标就可以用来做回归的预测(计算loss)
    
    step4. 计算损失
        > 分类损失基于交叉熵损失函数：loss(y_pred, y_label), 其中y_pred(8732, 21), y_label(8732,)，都是预测概率，评价的是两个预测概率分布的相关性。
        > 回归损失基于smoothl1损失函数：loss(y_pred, y_label), 其中y_pred(8732, 4), y_label(8732, 4), 都是坐标，评价的是类似于空间距离l2，但程度比l2稍微轻一点。
        注意： 这里分类损失的计算跟常规分类问题不同，常规分类loss(y_pred, y_label)，其中y_pred(b, 10), y_label(10,)，说明是以b张图片同时进行损失的多类别计算，每一行是一张图片的一个多分类问题。
        而在物体检测这里是以一张图片的b个anchors同时进行损失的多类别计算，每一行是一个anchor的一个多分类问题，再通过外循环进行多张图片的损失计算和汇总。
            
    """
    
    def __init__(self, 
                 input_size=(300,300),
                 num_classes=21,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 num_anchors=(4, 6, 6, 6, 4, 4),
                 anchor_size_ratio_range = (0.2, 0.9),
                 anchor_ratios = ([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 anchor_strides=(8, 16, 32, 64, 100, 300),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 neg_pos_ratio=3,
                 **kwargs):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.anchor_strides = anchor_strides
        self.featmap_sizes = [[ceil(input_size[0]/stride), ceil(input_size[1]/stride)] for stride in anchor_strides]
        self.target_means = target_means
        self.target_stds = target_stds 
        self.neg_pos_ratio = neg_pos_ratio
        
        # 创建分类分支，回归分支   
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.cls_convs.append(ClassHead(in_channels[i], num_anchors[i], num_classes))
            self.reg_convs.append(BboxHead(in_channels[i], num_anchors[i]))

        # 生成base_anchor所需参数
        n_featmap = len(in_channels)
        base_sizes, scales, ratios, centers = \
            get_base_anchor_params(input_size, anchor_size_ratio_range, 
                                   n_featmap, anchor_strides, anchor_ratios)
        # 创建anchor生成器
        self.anchor_generators = []
        for i in range(len(in_channels)):
            anchor_generator = AnchorGenerator(base_sizes[i], scales[i], 
                                               ratios[i], ctr=centers[i],    
                                               scale_major=False) 
            # 保留的anchor: 2*3的前4个(0-3), 2*5的前6个(0-5)
            keep_anchor_indices = range(0, len(ratios[i])+1)
            anchor_generator.base_anchors = anchor_generator.base_anchors[keep_anchor_indices]
            self.anchor_generators.append(anchor_generator)
        # 创建损失函数
        self.loss_cls = CrossEntropyLoss()
        self.loss_bbox = SmoothL1Loss()
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform", bias=0)
        
    def forward(self, x):
        cls_scores = []
        bbox_preds = []
        for feat, cls_conv, reg_conv in zip(x, self.cls_convs, self.reg_convs):
            cls_scores.append(cls_conv(feat))   # (6,)(b,-1,21)
            bbox_preds.append(reg_conv(feat))   # (6,)(b,-1,4)
        cls_scores = torch.cat(cls_scores, dim=1)        # (b,-1,21)
        bbox_preds = torch.cat(bbox_preds, dim=1)        # (b,-1,4)
        return dict(cls_scores=cls_scores, bbox_preds=bbox_preds) 
    
    def get_losses(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, cfg, **kwargs): # 用kwargs兼容ldmk
        """在训练时基于前向计算结果，计算损失
        cls_scores: (b,-1,21)
        bbox_preds: (b,-1,4)
        gt_bboxes: (b,)(m,4)
        gt_labels: (b,)(m,)
        cfg
        """
        num_imgs = len(gt_labels)
        multi_layer_anchors = []
        for i in range(len(self.featmap_sizes)):
            device = cls_scores.device
            anchors = self.anchor_generators[i].grid_anchors(
                self.featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_layer_anchors.append(anchors)  # (6,)(k, 4)
        multi_layer_anchors = torch.cat(multi_layer_anchors, dim=0)  # 堆叠(8732, 4)    
        anchor_list = [multi_layer_anchors for _ in range(num_imgs)]  # (b,) (s,4)
        # 计算target: None表示gt_landmarks=None
        target_result = get_anchor_target(anchor_list, gt_bboxes, gt_labels, None,
                                          cfg.assigner, cfg.sampler,
                                          self.target_means, self.target_stds)
        # 解析target
        bboxes_t, bboxes_w, labels_t, labels_w, _, _, num_pos, num_neg = target_result  # (b,-1,4)x2, (b,-1)x2
        
        # bbox回归损失
        pfunc = partial(self.loss_bbox, avg_factor=num_pos)
        loss_bbox = list(map(pfunc, bbox_preds, bboxes_t, bboxes_w))  # (b,)
        # cls分类损失
        loss_cls = list(map(self.loss_cls, cls_scores, labels_t))
        loss_cls = [loss_cls[i] * labels_w[i].float() for i in range(len(loss_cls))]  # (b,)(8732,)没有做loss缩减因为需要做负样本挖掘
        # cls loss的ohem
        pfunc = partial(ohem, neg_pos_ratio=self.neg_pos_ratio, avg_factor=num_pos)
        loss_cls = list(map(pfunc, loss_cls, labels_t))   # (b,)

        return dict(loss_cls = loss_cls, loss_bbox = loss_bbox)  # {(b,), (b,)} 每张图对应一个分类损失值和一个回归损失值。
             
        
#    def get_one_img_losses(self, cls_scores, bbox_preds, labels, label_weights, 
#                           bbox_targets, bbox_weights, num_total_samples, 
#                           neg_pos_ratio, cfg):
#        """计算单张图的分类回归损失，需要解决3个问题：
#        1. 为什么要引入负样本算损失？因为样本来自特征图，而特征图转换出来的子样本必然含有负样本，所以必须增加label=0的一类标签，作为21类做分类
#        2. 为什么正负样本比例是1:3？
#        3. 为什么损失值的平均因子是正样本个数？也就是单张图的分类损失和回归损失都用整个batch的正样本anchor个数进行了平均。
#        args:
#            cls_scores: (n_anchor, 21)
#            bbox_preds: (n_anchor, 4)
#            labels: (n_anchor,)
#            label_weights: (n_anchor, )
#            bbox_targets: (n_anchor, 4)
#            bbox_weights: (n_anchor, 4)
#            num_total_samples: 正样本数
#            cfg
#        """
#        
#        # 计算分类损失
#        loss_cls = F.cross_entropy(cls_scores, labels, reduction="none") # (8732)
#        loss_cls *= label_weights.float()  # (8732)
#        # OHEM在线负样本挖掘：提取损失中数值最大的前k个，并保证正负样本比例1:3 
#        loss_cls_pos, loss_cls_neg = ohem(loss_cls, labels, neg_pos_ratio)
#        # 规约分类损失
#        loss_cls_pos = loss_cls_pos.sum()
#        loss_cls_neg = loss_cls_neg.sum()
#        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
#        
#        # 计算回归损失
#        loss_reg = weighted_smooth_l1(bbox_preds, bbox_targets,
#                                      bbox_weights,
#                                      beta=cfg.loss_reg.beta,
#                                      avg_factor=num_total_samples)  # ()
#        return loss_cls, loss_reg
        
                
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg):
        """在测试时基于前向计算结果，计算bbox预测类别和预测坐标，此时前向计算后不需要算loss，直接计算bbox的预测
        Args:
            cls_scores(6,)(b,c,h,w): 按层分组
            bbox_preds(6,)(b,c,h,w):按层分组
            img_metas:()
            cfg:()
        """
        # 获得每层的grid-anchors
        featmap_sizes = [featmap.size() for featmap in cls_scores]
        num_imgs = len(img_metas)
        num_levels = len(cls_scores)
        multi_layer_anchors = []
        device = cls_scores[0].device
        for i in range(len(featmap_sizes)):
            anchors = self.anchor_generators[i].grid_anchors(featmap_sizes[i][2:], 
                                            self.anchor_strides[i], device=device)
            multi_layer_anchors.append(anchors)  # (6,)(k, 4)
        # 计算每张图的bbox预测
        bbox_results = [] 
        label_results = []
        for img_id in range(num_imgs):
            # 去掉batch这个维度，生成单图数据：(6,)(b,c,h,w)->(6,)(c,h,w)
            cls_score_per_img = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_per_img = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            
            img_shape = img_metas[img_id]['scale_shape']     # 这里传入scale_shape是用来clamp转换出来的bbox的坐标范围防止超出图片。
            scale_factor = img_metas[img_id]['scale_factor'] # scale factor直接判断
            bbox_result, label_result = self.get_one_img_bboxes(cls_score_per_img, bbox_pred_per_img,
                                                                multi_layer_anchors, img_shape,
                                                                scale_factor, cfg)
            bbox_results.append(bbox_result)
            label_results.append(label_result)
        return bbox_results, label_results  
    
    
    def get_one_img_bboxes(self, cls_scores, bbox_preds, multi_layer_anchors,
                           img_shape, scale_factor, cfg):
        """"对单张图进行预测：需要对每一特征图层分别处理，所以必须传入以level分组的数据
        1. cls_score概率化：采用softmax()函数
        3. 对bbox坐标逆变换delta2bbox()
        4. 进行nms过滤: 只是把空的bbox过滤(score<0.02)；同时去除重叠bbox；对置信度大小没有管控
        args:
            cls_scores: (6,)(c,h,w)
            bbox_preds: (6,)(c,h,w)
            multi_layer_anchors: (6,)(k,4)
            img_shape:
            scale_factor:
            cfg:
            rescale:
        """
        # 分别处理每一层特征层
        multi_layer_bboxes = []
        multi_layer_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds, multi_layer_anchors):
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes) # (c,h,w)->(-1, 21)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # 概率化
            scores = F.softmax(cls_score, dim=1)
            # 坐标化 
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means, self.target_stds, img_shape)
            #保存
            multi_layer_bboxes.append(bboxes)  
            multi_layer_scores.append(scores)
        # 堆叠 
        multi_layer_bboxes = torch.cat(multi_layer_bboxes)  # (6,)(m, 4) -> (8732, 4)
        multi_layer_scores = torch.cat(multi_layer_scores)  # (6,)(n, 21)-> (8732, 21)
        # 训练是基于scale之后img进行的，而最终得到的bbox需要在原图显示，所以要缩放回原图比例
        multi_layer_bboxes = multi_layer_bboxes / multi_layer_bboxes.new_tensor(scale_factor)  # (b,4)/(4,) = (b, 4)
        # 进行nms, 同时生成标签
        det_bboxes, det_labels = nms_operation(multi_layer_bboxes, multi_layer_scores, **cfg.nms)
        
        return det_bboxes, det_labels  # 坐标和置信度(k,5), 标签(k,)

    
        
# %%           
    
class RetinaHead(SSDHead):
    """retina head"""
    def __init__(self, 
                 input_size,
                 num_classes=21,
                 in_channels=256,
                 base_scale=4,
                 loss_cls_cfg=None,
                 loss_reg_cfg=None,
                 **kwargs):
        
        super().__init__()
        

# %%
if __name__ == "__main__":
    '''base_anchor的标准数据
    [[-11., -11.,  18.,  18.],[-17., -17.,  24.,  24.],[-17.,  -7.,  24.,  14.],[ -7., -17.,  14.,  24.]]
    [[-22., -22.,  37.,  37.],[-33., -33.,  48.,  48.],[-34., -13.,  49.,  28.],[-13., -34.,  28.,  49.],[-44.,  -9.,  59.,  24.],[ -9., -44.,  24.,  59.]]
    [[-40., -40.,  70.,  70.],[-51., -51.,  82.,  82.],[-62., -23.,  93.,  54.],[-23., -62.,  54.,  93.],[-80., -16., 111.,  47.],[-16., -80.,  47., 111.]]
    [[ -49.,  -49.,  112.,  112.],[ -61.,  -61.,  124.,  124.],[ -83.,  -25.,  146.,   88.],[ -25.,  -83.,   88.,  146.],[-108.,  -15.,  171.,   78.],[ -15., -108.,   78.,  171.]]
    [[ -56.,  -56.,  156.,  156.],[ -69.,  -69.,  168.,  168.],[-101.,  -25.,  200.,  124.],[ -25., -101.,  124.,  200.]]
    [[ 18.,  18., 281., 281.],[  6.,   6., 293., 293.],[-37.,  57., 336., 242.],[ 57., -37., 242., 336.]]
    '''
    
#    import sys, os
#    path = os.path.abspath("../utils")
#    if not path in sys.path:
#        sys.path.insert(0, path)
    from utils.visualization import vis_bbox
    name = 'test1'
    
    if name == 'test1':  # 检测生成的base anchor是否正确
        base_sizes, anchor_scales, anchor_ratios, centers  = \
            get_base_anchor_params(img_size=300, ratio_range=[0.2, 0.9], 
                                   n_featmap=6, 
                                   strides=[8,16,32,64,128,300], 
                                   ratios=[[2],[2, 3],[2, 3],[2, 3],[2],[2]])
        anchor_generators = []
        for i in range(len(base_sizes)):
            anchor_generator = AnchorGenerator(base_sizes[i],
                                               anchor_scales[i],
                                               anchor_ratios[i],
                                               ctr=centers[i],
                                               scale_major=False)
            
            base_anchors = anchor_generator.base_anchors
            vis_bbox(base_anchors)
            anchor_generators.append(anchor_generator)
            
      
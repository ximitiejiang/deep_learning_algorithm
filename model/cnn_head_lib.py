#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:15:42 2019

@author: ubuntu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.module_factory import registry
from utils.weight_init import xavier_init


@registry.register_module    
class SSDHead(nn.Module):
    """分类回归头：
    1. 分类回归头的工呢过：保持尺寸w,h不变，变换层数进行层数匹配，把特征金字塔的每一级输出层变换成num_anchor
    其中分类层：输出层数 = 目标分类指标数量 = 类别数*每个特征像素映射到原图后上面放置的anchor个数
        这里anchor个数在特征金字塔不同层不同，分别是(4,6,6,6,4,4)
    其中回归层：输出层数 = 目标回归指标数量 = 回归坐标数*每个特征像素映射到原图后上面放置的anchor个数
           -----------------
          /                 \
        [cls]               [reg]
        3x3(512 to 81*4)    3x3(512 to 4*4)
        3x3(1024 to 81*6)   3x3(1024 to 4*6)
        3x3(512 to 81*6)    3x3(512 to 4*6)
        3x3(256 to 81*6)    3x3(256 to 4*6)
        3x3(256 to 81*4)    3x3(256 to 4*4) 
        3x3(256 to 81*4)    3x3(256 to 4*4) 
    
    2. anchor生成机制：
    其中anchor的个数(4,6,6,6,4,4)是根据经验
    
    
    """
    
    def __init__(self, 
                 input_size=300,
                 num_classes=21,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 num_anchors=(4, 6, 6, 6, 4, 4),
                 anchor_strides=(8, 16, 32, 64, 100, 300),

                 target_means=(.0, .0, .0, .0),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 **kwargs): # 增加一个多余变量，避免修改cfg, 里边有一个type变量没有用
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.anchor_strides = anchor_strides
        self.target_means = target_mean
        self.target_stds = target_stds 
        # 创建分类分支，回归分支
        cls_convs = []
        reg_convs = []
        for i in range(len(in_channels)):
            cls_convs.append(
                    nn.Conv2d(in_channels[i], num_anchors[i], kernel_size=3, padding=1))
            reg_convs.append(
                    nn.Conv2d(in_channels[i], num_anchors[i], kernel_size=3, padding=1))
        self.cls_convs = nn.ModuleList(cls_convs) # 由于6个convs是并行分别处理每一个特征层，所以不需要用sequential
        self.reg_convs = nn.ModuleList(reg_convs)
        
        # 生成anchor所需标准参数
        # base sizes：表示一组anchor的基础大小，然后乘以scale(比例)，变换宽高比(ratio)
        min_sizes = [30, 60, 111, 162, 213, 264]
        max_sizes = [60, 111, 162, 213, 264, 315]
        base_sizes = min_sizes
        # strides: 表示两组anchor中心点的距离
        strides = anchor_strides
        # centers: 表示每组anchor的中心点坐标
        centers = []
        for stride in strides:
            centers.append(((stride - 1) / 2., (stride - 1) / 2.))
        # scales: 表示anchor基础尺寸的放大倍数
        scales = []       
        for max_size, min_size in zip(max_sizes, min_sizes):
            scales.append([1., np.sqrt(max_size / min_size)])  # ssd定义2种scale(1,sqrt(k))
        # ratios：表示anchor的高与宽的比例
        ratios = ([1, 1/2, 2], [1, 1/2, 2, 1/3, 3], [1, 1/2, 2, 1/3, 3], [1, 1/2, 2, 1/3, 3], [1, 1/2, 2], [1, 1/2, 2])
        # 生成anchor
        self.anchor_generators = []
        for base_size, scale, ratio, ctr in zip(base_sizes, scales, ratios, centers):
            anchor_generator = AnchorGenerator(base_size, scale, ratio, scale_major=False, ctr=ctr)
            # 截取一定个数的anchors作为base anchor
            keep_anchor_indics = range(0, len(ratio)+1)   # 保留的anchor: 2*3的前(0-3), 2*5的前(0-5)
            anchor_generator.base_anchors = anchor_generator.base_anchors[keep_anchor_indics]
        
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform", bias=0)
        
    def forward(self, x):
        cls_scores = []
        bbox_preds = []
        for i, feat in enumerate(x):
            cls_scores.append(self.cls_convs[i](feat))
            bbox_preds.append(self.reg_convs[i](feat))
        return cls_scores, bbox_preds 
    
    def get_losses(self, cls_scores, bbox_preds, 
                   gt_bboxes, gt_labels, 
                   img_metas, cfg):
        """在训练时基于前向计算结果，计算损失"""
        # 获得各个特征图尺寸: (6,)-(38,38)(19,19)(10,10)(5,5)(3,3)(1,1)
        featmap_sizes = [featmap.size() for featmap in cls_scores]
        num_imgs = len(img_metas)
        # 生成每个特征图的grid anchors
        multi_layers_anchors = []
        for i in range(len(featmap_sizes)):
            anchors = self.anchor_generators.grid_anchors(featmap_sizes[i], self.anchor_strides[i])
            multi_layers_anchors.append(anchors)  # (6,)(k, 4)
        
        num_level_anchors = [an.numel() for an in multi_layers_anchors]
        multi_layers_anchors = torch.cat(multi_layers_anchors, dim=0)  # (8732, 4)    
        anchor_list = [multi_layers_anchors for _ in range(len(img_metas))]  # (n_imgs,) (s,4)
        
        # TODO: 没有采用valid_flag_list
        (all_bbox_targets,     # (6,) (4, 5776, 4)
         all_bbox_weights,     # (6,) (4, 5776, 4)
         all_labels,           # (6,) (4, 5776)  
         all_label_weights,    # (6,) (4, 5776)  
         num_total_pos, 
         num_total_neg) = self.get_anchor_target(anchor_list, gt_bboxes, img_metas, cfg.assigner, num_level_anchors)
        
        # 计算损失:
        # 先组合一个batch所有图片的数据
        all_cls_scores = [s.permute(0,2,3,1).reshape(num_imgs, -1, self.cls_out_channels) for s in cls_scores]    
        all_cls_scores = torch.cat(all_cls_scores, dim=1)        # 从(6,)(4,84,38,38)...到(4, 8732, 21)
#        all_labels = torch.cat(all_labels, dim=-1)               # 从(6,)(4,5776)到(4, 8732)
#        all_label_weights = torch.cat(all_label_weights, dim=-1) # 从(6,)(4,5776)到(4, 8732)
        
        all_bbox_preds = [p.permute(0,2,3,1).reshape(num_imgs, -1, 4) for p in bbox_preds]
        all_bbox_preds = torch.cat(all_bbox_preds, dim=1)        # 从(6,)(4,16,38,38)...到(4, 8732, 4)
#        all_bbox_targets = torch.cat(all_bbox_targets, dim=1)    # 从(6,) (4, 5776, 4)到(4, 8732, 4)
#        all_bbox_weights = torch.cat(all_bbox_weihts, dim=1)     # 从(6,) (4, 5776, 4)到(4, 8732, 4)
        
        all_loss_cls = []
        all_loss_reg = []
        for _ in range(num_imgs):  # 分别计算每张图的损失
            # 计算分类损失
            loss_cls = F.cross_entropy(all_cls_score[i], all_labels[i], reduction="none") * all_label_weights[i] # {(8732,21),(8732,)} *(8732,4)
            
            # OHEM在线负样本挖掘：提取损失中数值最大的前k个，并保证正负样本比例1:3 
            # (这样既保证正负样本平衡，也保证对损失贡献大的负样本被使用)
            pos_inds = np.where(all_labels[i] > 0)
            neg_inds = np.where(all_labels[i] == 0)
            num_pos_samples = pos_inds.shape[0]
            num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
            topk_loss_cls = loss_cls[neg_inds].sort()[0][num_neg_samples]
            
            # 计算平均损失
#            num_total_samples = 
#            loss_cls_pos_sum = 
#            loss_cls_neg_sum = 
            loss_cls = (loss_cls_pos_sum + loss_cls_neg_sum) / num_total_samples
            
            # 计算回归损失
            loss_reg = weighted_smoothl1(all_bbox_preds[i],
                                         all_bbox_targets[i],
                                         all_bbox_weights[i],
#                                         beta=,
                                         avg_factor=num_total_samples)
        
        return dict(loss_cls=all_loss_cls, loss_reg = all_loss_reg)
    
    def get_anchor_target(self, anchor_list, gt_bboxes_list, img_metas_list, assign_cfg, gt_labels_list, num_level_anchors):
        """计算一个batch的多张图片的anchor target
        Input:
            anchor_list: (n_imgs, )(s, 4)
            gt_bboxes_list: (n_imgs, )(k, 4)
            img_metas_list： (n_imgs, )(dict)
            gt_labels_list: (n_imgs, )(m, )
        """
        # TODO: 放在哪个模块里边比较合适
        all_labels = []
        all_label_weights = []
        all_bbox_targets = []
        all_bbox_weights = []
        
        all_pos_inds_list = []
        all_neg_inds_list = []
        for i in range(len(img_metas_list)): # 对每张图分别计算
            # 1.指定: 指定每个anchor是正样本还是负样本(基于跟gt进行iou计算)
            bbox_assigner = MaxIouAssigner(**assign_cfg)
            assign_result = bbox_assigner.assign(anchor_list[i], gt_bboxes_list[i], gt_labels_list[i])
            # 2.采样： 采样一定数量的正负样本:通常用于预防正负样本不平衡
            #　但在SSD中没有采样只是区分了一下正负样本，所以这里只是一个假采样。(正负样本不平衡是通过最后loss的OHEM完成)
            bbox_sampler = PseudoSampler()
            sampling_result = bbox_sampler.sample(assign_result, anchor_list[i], gt_bboxes_list[i])
            assigned_gt_inds, assigned_gt_labels, ious = assign_result  # (m,), (m,) 表示anchor的身份, anchor对应的标签，[0,1,0,..2], [0,15,...18]
            pos_assigned_gt_inds = assigned_gt_inds[assigned_gt_inds > 0] - 1 #从1～k变换到0~k-1
            pos_inds, neg_inds = sampling_result  # (k,), (j,) 表示正/负样本的位置号，比如[7236, 7249, 8103], [0,1,2...8104..]
            
            # 3. 计算每个anchor的target：基于找到的正样本，得到bbox targets, bbox_weights     
            bbox_targets = torch.zeros_like(anchor_list[i])
            bbox_weights = torch.zeros_like(anchor_list[i])
            labels = torch.zeros(len(anchor_list[i]), dtype=torch.long)
            label_weights = torch.zeros(len(anchor_list[i]), dtype=torch.long)
            
            pos_bboxes = anchor_list[i][pos_inds]                # (k,4)获得正样本bbox
            pos_gt_bboxes = gt_bboxes_list[i][pos_assigned_gt_inds]  # (k,4)获得正样本bbox对应的gt bbox坐标
            pos_bbox_target = bbox2delta(pos_bboxes, pos_gt_bboxes)  
            
            bbox_targets[pos_inds] = pos_bbox_targets
            bbox_weights[pos_inds] = 1
            # 得到labels, label_weights
            labels[pos_inds] = gt_labels_list[i][pos_assigned_gt_inds]
            label_weights[pos_inds] = 1  # 这里设置正负样本权重=1， 如果有需要可以提高正样本权重
            label_weights[neg_inds] = 1
            
            # batch图片targets汇总
            all_bbox_targets.append(bbox_targets)   # (n_img, ) (n_anchor, 4)
            all_bbox_weights.append(bbox_weights)   # (n_img, ) (n_anchor, 4)
            all_labels.append(labels)                # (n_img, ) (n_anchor, )
            all_label_weights.append(label_weights)  # (n_img, ) (n_anchor, )
            all_pos_inds_list.append(pos_inds)       # (n_img, ) (k, )
            all_neg_inds_list.append(neg_inds)       # (n_img, ) (j, )
            
        # 4. 对targets数据进行变换，按照特征图个数把每个数据分成6份，把多张图片的同尺寸特征图的数据放一起，统一做loss
        all_bbox_targets = torch.stack(all_bbox_targets, dim=0)   # (n_img, n_anchor, 4)
        all_bbox_weights = torch.stack(all_bbox_weights, dim=0)   # (n_img, n_anchor, 4)
        all_labels = torch.stack(all_labels, dim=0)               # (n_img, n_anchor)
        all_label_weights = torch.stack(all_label_weights, dim=0) # (n_img, n_anchor)
        
        num_total_pos = sum([inds.numel() for inds in all_pos_inds_list])
        num_total_neg = sum([inds.numel() for inds in all_neg_inds_list])
        
#        def distribute_to_level(target, num_level_anchors):
#            """把(n_img, n_anchor, 4)或(n_img, n_anchor, )的数据分配到每个level去，
#             变成(n_level,) (n_imgs, n_anchor)或(n_level,) (n_imgs, n_anchor)"""
#            distributed = []
#            start=0
#            for n in num_level_anchors:
#                end = start + n
#                distributed.append(target[:, start:end])
#                start = end
#        
#        distribute = False
#        if distribute:
#            all_bbox_targets = distribute_to_level(all_bbox_targets, num_level_anchors)  # (6,) (4, 5776, 4)
#            all_bbox_weights = distribute_to_level(all_bbox_weights, num_level_anchors)  # (6,) (4, 5776, 4)
#            all_labels = distribute_to_level(all_labels, num_level_anchors)              # (6,) (4, 5776)  
#            all_label_weights = distribute_to_level(all_label_weights, num_level_anchors)# (6,) (4, 5776)

        return (all_bbox_targets, all_bbox_weights, all_labels, all_label_weights, num_total_pos, num_total_neg)
            
                
    def get_bboxes(self, cls_scores, bbox_preds):
        """在测试时基于前向计算结果，计算bbox预测值，此时前向计算后不需要算loss，直接算bbox"""
        pass

# %%
class AnchorGenerator():
    """生成base anchors和grid anchors"""
    def __init__(self, base_size, scales, ratios, scale_major=False, ctr=None):
        self.base_size = base_size
        self.scales = np.array(scales)
        self.ratios = np.array(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        
        self.base_anchors = self.get_base_anchors()
    
    def get_base_anchors(self): 
        """生成单个特征图的base anchors"""
        w, h = self.base_size, self.base_size
        # 准备中心点坐标
        if self.ctr is None:
            x_ctr, y_ctr = 0.5 * (w - 1), 0.5 * (h - 1) 
        else:
            x_ctr, y_ctr = self.ctr
        # 准备宽高的比例: ratio=h/w=2则h=sqrt(2)*h0, w=1/sqrt(2) * w0
        h_ratios = np.sqrt(self.ratios) #(n,)
        w_ratios = 1 / h_ratios         #(n,)
        # 计算变换后的w', h'
        if self.scale_major:
            w_new = (w * w_ratios[:, None] *self.scales[None, :]).reshape(-1)    # (n,1)*(1,m)->(n,m)->(n*m,)
            h_new = (h * h_ratios[:, None] *self.scales[None, :]).reshape(-1)
        else:
            w_new = (w * self.scales[:, None] * w_ratios[None, :]).reshape(-1)    # (m,1)*(1,n)->(m,n)->(m*n,)
            h_new = (h * self.scales[:, None] * h_ratios[None, :]).reshape(-1)
        # 计算坐标xmin,ymin,xmax,ymax
        base_anchors = np.stack([x_ctr - 0.5 * (w_new - 1), 
                                 y_ctr - 0.5 * (h_new - 1),
                                 x_ctr + 0.5 * (w_new - 1), 
                                 y_ctr + 0.5 * (h_new - 1)], axis=-1).round()  # (m*n, 4))
        
        return torch.tensor(base_anchors)
    
    def grid_anchors(self, featmap_size, stride):
        """生成单个特征图的网格anchors"""
        #TODO: 检查是否要送入device
        base_anchors = self.base_anchors #(k, 4)
        # 生成原图上的网格坐标
        h, w = featmap_size
        x = np.arange(0, w) * stride  # (m,)
        y = np.arange(0, h) * stride  # (n,)
        xx = np.tile(x, (len(y),1)).reshape(-1)                # (m*n,)
        yy = np.tile(y.reshape(-1,1), (1, len(x))).reshape(-1) # (m*n,)
        # 基于网格坐标生成(xmin, ymin, xmax, ymax)的坐标平移矩阵
        shifts = np.stack([xx, yy, xx, yy], axis=-1)  # (m*n, 4)
        shifts = torch.tensor(shifts)
        # 平移anchors: 相当于该组anchors跟每个平移坐标进行相加，
        # 也就相当于要取每一行坐标跟一组anchor运算，所以用坐标插入空轴而不是用anchor插入空轴
        all_anchors = base_anchors + shifts[:, None, :]   #(b,4)+(k,1,4)->(k, b, 4)
        all_anchors = all_anchors.reshape(-1, 4)  # (k*b, 4)
        return all_anchors
    
# %%    
class MaxIouAssigner():
    """用于指定每个anchor的身份是正样本还是负样本：基于anchor跟gt_bboxes的iou计算结果进行指定"""
    def __init__(self, pos_iou_thr, neg_iou_thr, min_pos_iou=0):
        self.pos_iou_thr = pos_iou_thr     #正样本阈值：大于该值则为正样本
        self.neg_iou_thr = neg_iou_thr     #负样本阈值：小于该值则为负样本
        self.min_pos_iou = min_pos_iou     #最小正样本阈值： 
        
    def assign(self, anchors, gt_bboxes, gt_labels):
        # 计算ious
        ious = get_ious(gt_bboxes, anchors)  # (m,4)(n,4)->(m,n)
        anchor_maxiou_for_all_gt = ious.max(axis=0)          # (n,)
        anchor_maxiou_idx_for_all_gt = ious.argmax(axis=0)   # (n,) 0~n_gt
        
        #gt_maxiou_for_all_anchor = ious.max(axis=1)          # (m,)
        gt_maxiou_idx_for_all_anchor = ious.argmax(axis=1)   # (m,)
        
        # 基于规则指定每个anchor的身份, 创建anchor标识变量，先指定所有anchor为-1
        # 然后设置负样本=0，正样本=idx+1>0
        num_anchors = ious.shape[1]
        assigned_gt_inds = np.full((num_anchors, ), -1)      # (n, )
        # 小于负样本阈值，则设为0
        neg_idx = (anchor_maxiou_for_all_gt < self.neg_iou_thr) & (anchor_maxiou_for_all_gt >=0)
        assigned_gt_inds[neg_idx] = 0
        # 大于正样本阈值，则设为对应gt的index + 1 (>0)也代表gt的编号
        pos_idx = (anchor_maxiou_for_all_gt > self.pos_iou_thr) & (anchor_maxiou_for_all_gt <1)
        assigned_gt_inds[pos_idx] = anchor_maxiou_idx_for_all_gt[pos_idx] + 1 # 从0~k-1变到1～k,该值就代表了第几个gt   
        # 每个gt所对应的最大iou的anchor也设置为index + 1(>0)也代表gt的编号
        # 这样确保每个gt至少有一个anchor对应
        for i, anchor_idx in enumerate(gt_maxiou_idx_for_all_anchor):
            assigned_gt_inds[anchor_idx] = i + 1   # 从0~k-1变到1～k,该值就代表了第几个gt 
        
        # 转换正样本的标识从1~indx+1为真实gt_label
        assigned_gt_labels = np.zeros((num_anchors, ))     # (n, )
        for i, assign in enumerate(assigned_gt_inds):
            if assign > 0:
                label = gt_labels[assign-1]
                assigned_gt_labels[i] = label
        
        return [assigned_gt_inds, assigned_gt_labels, ious] # [(n,), (n,), (m,n)] 
            
    
class PseudoSampler():
    def __init__(self):
        pass
    
    def sample(self, assign_result, anchor_list, gt_bboxes):
        # 提取正负样本的位置号
        pos_inds = np.where(assign_result[0] > 0)[0]  #
        neg_inds = np.where(assign_result[0] == 0)[0]
        
#        pos_bboxes = 
        
        return [pos_inds, neg_inds]
        

def bbox_ious(bboxes1, bboxes2):
    """用于计算两组bboxes中每2个bbox之间的iou(包括所有组合，而不只是位置对应的bbox)
    bb1(m, 4), bb2(n, 4), 假定bb1是gt_bbox，则每个gt_bbox需要跟所有anchor计算iou，
    也就是提取每一个gt，因此先从bb1也就是bb1插入轴，(m,1,4),(n,4)->(m,n,4)，也可以先从bb2插入空轴则得到(n,m,4)"""
    # 在numpy环境操作(也可以用pytorch)
    bb1 = bboxes1.numpy()
    bb2 = bboxes2.numpy()
    # 计算重叠区域的左上角，右下角坐标
    xymin = np.max(bb1[:, None, :2] , bb2[:, :2])  # (m,2)(n,2) -> (m,1, 2)(n,2) -> (m,n,2)
    xymax = np.min(bb1[:, 2:] , bb2[:, None, 2:])  # (m,2)(n,2) -> (m,1, 2)(n,2) -> (m,n,2)
    # 计算重叠区域w,h
    wh = xymax - xymin # (m,n,2)-(m,n,2) = (m,n,2)
    # 计算重叠面积和两组bbox面积
    area = wh[:, :, 0] * wh[:, :, 1] # (m,n)
    area1 = (bb1[:, 2] - bb1[:, 0]) * (bb1[:, 3] - bb1[:, 1]) # (m,)*(m,)->(m,)
    area2 = (bb2[:, 2] - bb2[:, 0]) * (bb2[:, 3] - bb2[:, 1]) # (n,)*(n,)->(n,)
    # 计算iou
    ious = area / (area1 + area2[:,None,:] - area)     #(m,n) /[(m,)+(1,n)-(m,n)] -> (m,n) / (m,n)
    
    return ious  # (m,n)


def bbox2delta(prop, gt):
    """把proposal的anchor(k, 4)转化为相对于gt(k,4)的变化dx,dy,dw,dh
    基本逻辑：由前面的卷积网络可以得到预测xmin,ymin,xmax,ymax，并转化成px,py,pw,ph.
    此时存在一种变换dx,dy,dw,dh，可以让预测值变成gx',gy',gw',gh'且该值更接近gx,gy,gw,gh
    所以目标就变成找到dx,dy,dw,dh，寻找的方式就是dx=(gx-px)/pw, dy=(gy-py)/ph, dw=log(gw/pw), dh=log(gh/ph)
    因此卷积网络前向计算每次都得到xmin/ymin/xmax/ymax经过head转换成dx,dy,dw,dh，力图让loss最小使这个变换
    最后测试时head计算得到dx,dy,dw,dh，就可以通过delta2bbox()反过来得到xmin,ymin,xmax,ymax
    """
    # 把xmin,ymin,xmax,ymax转换成x_ctr,y_ctr,w,h
    px = (prop[...,0] + prop[...,2]) * 0.5
    py = (prop[...,1] + prop[...,3]) * 0.5
    pw = (prop[...,2] - prop[...,0]) + 1.0  
    ph = (prop[...,3] - prop[...,1]) + 1.0
    
    gx = (gt[...,0] + gt[...,2]) * 0.5
    gy = (gt[...,1] + gt[...,3]) * 0.5
    gw = (gt[...,2] - gt[...,0]) + 1.0  
    gh = (gt[...,3] - gt[...,1]) + 1.0
    # 计算dx,dy,dw,dh
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1) # (n, 4)
    # 归一化
    means = [0,0,0,0]    # (4,)
    std = [0.1,0.1,0.2,0.2]      # (4,)
    deltas = (deltas - means[None, :]) / std[None, :]    # (n,4)-(1,4) / (1,4) -> (n,4) / (1,4) -> (n,4) 
    

def delta2bbox():
    pass    

# %%
if __name__ == "__main__":
    """base_anchor的标准数据
    [[-11., -11.,  18.,  18.],[-17., -17.,  24.,  24.],[-17.,  -7.,  24.,  14.],[ -7., -17.,  14.,  24.]]
    [[-22., -22.,  37.,  37.],[-33., -33.,  48.,  48.],[-34., -13.,  49.,  28.],[-13., -34.,  28.,  49.],[-44.,  -9.,  59.,  24.],[ -9., -44.,  24.,  59.]]
    [[-40., -40.,  70.,  70.],[-51., -51.,  82.,  82.],[-62., -23.,  93.,  54.],[-23., -62.,  54.,  93.],[-80., -16., 111.,  47.],[-16., -80.,  47., 111.]]
    [[ -49.,  -49.,  112.,  112.],[ -61.,  -61.,  124.,  124.],[ -83.,  -25.,  146.,   88.],[ -25.,  -83.,   88.,  146.],[-108.,  -15.,  171.,   78.],[ -15., -108.,   78.,  171.]]
    [[ -56.,  -56.,  156.,  156.],[ -69.,  -69.,  168.,  168.],[-101.,  -25.,  200.,  124.],[ -25., -101.,  124.,  200.]]
    [[ 18.,  18., 281., 281.],[  6.,   6., 293., 293.],[-37.,  57., 336., 242.],[ 57., -37., 242., 336.]]
    """
    
    import sys, os
    path = os.path.abspath("../utils")
    if not path in sys.path:
        sys.path.insert(0, path)
    
    head = SSDHead()
    
    """广播机制的应用: 前提是两个变量从右往左，右边的对应轴size要相同，或者其中一个变量size=0或1
        对其中一个变量插入一个轴，就相当于对他提取每一行，并广播成另一个变量的形状"""
    a = np.ones((3,4))
    b = np.ones((10,4))
    result = a + b[:, None, :]
    
    """广播机制的应用2: 有3个角色分别有各自的攻击力和防御力，各自攻击100个目标分别获取攻击防御力加成"""
    roles = np.array([[1,2],[3,2],[4,1]])           # (3,2)
    objects = np.random.randint(1,10, size=(100,2)) # (100,2)
    result = roles + objects[:, None, :]    # (3,2)+(100,1,2)->(100,3,2)
    
    
        
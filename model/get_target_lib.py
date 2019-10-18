#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:07:24 2019

@author: ubuntu
"""
import torch
from functools import partial
from model.anchor_assigner_sampler_lib import MaxIouAssigner, PseudoSampler
from model.bbox_regression_lib import bbox2delta, bbox2lrtb, landmark2delta


def get_anchor_target(anchor_list, gt_bboxes_list, gt_labels_list, gt_landmarks_list,
                      assigner_cfg, sampler_cfg, means, stds):
    """简化get anchor target的写法：基于batch的gt数据(b,)，让anchors能够匹配到合适的target(这个target可能是gt，可能是背景)
    """
    if gt_landmarks_list is None:
        gt_landmarks_list = [None for _ in anchor_list]
    pfunc = partial(anchor_match_target, assigner_cfg=assigner_cfg, sampler_cfg=sampler_cfg, means=means, stds=stds)
    targets = list(map(pfunc, anchor_list, gt_bboxes_list, gt_labels_list, gt_landmarks_list))  # (b,) (6,)生成每张图的target

    bboxes_t = torch.stack([result[0] for result in targets])   # (b,-1,4)
    bboxes_w = torch.stack([result[1] for result in targets])   # (b,-1,4)
    labels_t = torch.stack([result[2] for result in targets])   # (b,-1)
    labels_w = torch.stack([result[3] for result in targets])   # (b,-1)
    num_pos = sum([result[4].numel() for result in targets])    # (m,)
    num_neg = sum([result[5].numel() for result in targets])    # (n,)  m+n = b * 8732
    
    return bboxes_t, bboxes_w, labels_t, labels_w, num_pos, num_neg
    

def anchor_match_target(anchors, gt_bboxes, gt_labels, gt_ldmks,
                        assigner_cfg, sampler_cfg, means, stds):
    """核心程序: 对单张图的anchor进行目标匹配。
    让每个anchor都能匹配到合适的target，如果匹配的target是gt就获得对应gt的数据(bbox,label,ldmk)
    如果匹配的target是背景就置0.
    """
    # 1.指定: 指定每个anchor是正样本还是负样本(通过让anchor跟gt bbox进行iou计算来评价，得到每个anchor的)
    bbox_assigner = MaxIouAssigner(**assigner_cfg.params)
    assign_result = bbox_assigner.assign(anchors, gt_bboxes, gt_labels)
    assigned_gt_inds, assigned_gt_labels, ious = assign_result  # 前者(8732,)表示anchor的身份是第几个gt(从1开始，1则表示第1个gt class, 0表示负样本)
                                                                # 后者(8732,)表示anchor的身份标签为0~20
    # 2. 采样： 采样一定数量的正负样本, 通常用于预防正负样本不平衡
    #　但在SSD中没有采样只是区分了一下正负样本，所以这里只是一个假采样。(正负样本不平衡是通过最后loss的OHEM完成)
    bbox_sampler = PseudoSampler(**sampler_cfg.params)
    sampling_result = bbox_sampler.sample(*assign_result, anchors, gt_bboxes)
    pos_inds, neg_inds = sampling_result  # (k,), (j,) 表示anchor中第几个anchor为正样本，第几个样本为负样本
    
    # 3. 初始化target    
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(len(anchors), dtype=torch.long)       # 借用anchors的device
    label_weights = anchors.new_zeros(len(anchors), dtype=torch.long)# 借用anchors的device
    # 4. 把正样本 bbox坐标转换成delta坐标并填入
    pos_bboxes = anchors[pos_inds]                   # (k,4)表示从8000多个anchors里边提取出对应正样本的anchor作为bbox
    pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1 # 表示正样本对应的label也就是gt_bbox是第几个，并且从第0个开始(减1所以从1-n变成0-n-1)
    pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]  # (k,4)获得正样本bbox对应的gt bbox坐标
    
#    ldmk_targets = anchors.new_zeros(len(anchors), 10, dtype=torch.float32)
#    ldmk_weights = anchors.new_zeros(len(anchors), 10, dtype=torch.float32)
#    pos_gt_ldmks = gt_ldmks[pos_assigned_gt_inds]
#    pos_ldmk_targets = landmark2delta(pos_bboxes, pos_gt_ldmks, means, stds)
#    ldmk_targets[pos_inds] = pos_ldmk_targets
#    ldmk_weights[pos_inds] = 1
    
    pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, means, stds)  # 
    bbox_targets[pos_inds] = pos_bbox_targets
    bbox_weights[pos_inds] = 1
    # 5. 把正样本labels填入
    labels[pos_inds] = gt_labels[pos_assigned_gt_inds] # 获得正样本对应label
    label_weights[pos_inds] = 1  # 这里设置正负样本权重都为=1， 如果有需要可以提高正样本权重
    label_weights[neg_inds] = 1
      
    return (bbox_targets, bbox_weights, 
            labels, label_weights, 
#            ldmk_targets, ldmk_weights,
            pos_inds, neg_inds)    

    
#
#def get_anchor_target1(anchor_list, gt_bboxes_list, gt_labels_list,
#                      img_metas_list, assigner_cfg, sampler_cfg, 
#                      num_level_anchors, target_means, target_stds):
#        """在ssd算法中计算一个batch的多张图片的anchor target，也就是对每个anchor定义他所属的label和坐标：
#        其中如果是正样本则label跟对应bbox一致，坐标跟对应bbox也一致；如果是负样本则label
#        Input:
#            anchor_list: (b, )(s, 4)
#            gt_bboxes_list: (b, )(k, 4)
#            gt_labels_list: (b, )(k, )
#            img_metas_list： (b, )(dict)
#        return:
#            
#        """
#        # 初始化
#        all_labels = []
#        all_label_weights = []
#        all_bbox_targets = []
#        all_bbox_weights = []
#        
#        all_pos_inds_list = []
#        all_neg_inds_list = []
#        # 循环对每张图分别计算anchor target
#        for i in range(len(img_metas_list)):        
#            bbox_targets, bbox_weights, labels, label_weights, pos_inds, neg_inds = \
#                get_one_img_anchor_target(anchor_list[i],
#                                          gt_bboxes_list[i],
#                                          gt_labels_list[i],
#                                          img_metas_list[i],
#                                          assigner_cfg,
#                                          sampler_cfg,
#                                          target_means,
#                                          target_stds)
#            # batch图片targets汇总
#            all_bbox_targets.append(bbox_targets)   # (b, ) (n_anchor, 4)
#            all_bbox_weights.append(bbox_weights)   # (b, ) (n_anchor, 4)
#            all_labels.append(labels)                # (b, ) (n_anchor, )
#            all_label_weights.append(label_weights)  # (b, ) (n_anchor, )
#            all_pos_inds_list.append(pos_inds)       # (b, ) (k, )
#            all_neg_inds_list.append(neg_inds)       # (b, ) (j, )
#            
#        # 对targets数据进行变换，把多张图片的同尺寸特征图的数据放一起，统一做loss
#        all_bbox_targets = torch.stack(all_bbox_targets, dim=0)   # (b, n_anchor, 4)
#        all_bbox_weights = torch.stack(all_bbox_weights, dim=0)   # (b, n_anchor, 4)
#        all_labels = torch.stack(all_labels, dim=0)               # (b, n_anchor)
#        all_label_weights = torch.stack(all_label_weights, dim=0) # (b, n_anchor)
#        
#        num_batch_pos = sum([inds.numel() for inds in all_pos_inds_list])
#        num_batch_neg = sum([inds.numel() for inds in all_neg_inds_list])
#
#        return (all_bbox_targets, all_bbox_weights, all_labels, all_label_weights, num_batch_pos, num_batch_neg)
#
#
#def get_one_img_anchor_target(anchors, gt_bboxes, gt_labels, img_metas, 
#                              assigner_cfg, sampler_cfg, target_means, target_stds):
#    """针对单张图的anchor target计算： 本质就是把正样本的坐标和标签作为target
#    提供一种挑选target的思路：先iou筛选出正样本(>0)和负样本(=0)，去掉无关样本(-1)
#    然后找到对应正样本anchor, 换算出对应bbox和对应label
#    
#    args:
#        anchors (s, 4)
#        gt_bboxes (k, 4)
#        gt_labels (k, )
#        img_metas (dict)
#        assigner_cfg (dict)
#        sampler_cfg (dict)
#    """
#    # 1.指定: 指定每个anchor是正样本还是负样本(通过让anchor跟gt bbox进行iou计算来评价，得到每个anchor的)
#    bbox_assigner = MaxIouAssigner(**assigner_cfg.params)
#    assign_result = bbox_assigner.assign(anchors, gt_bboxes, gt_labels)
#    assigned_gt_inds, assigned_gt_labels, ious = assign_result  # (m,), (m,) 表示anchor的身份, anchor对应的标签，[0,1,0,..2], [0,15,...18]
#    
#    # 2. 采样： 采样一定数量的正负样本, 通常用于预防正负样本不平衡
#    #　但在SSD中没有采样只是区分了一下正负样本，所以这里只是一个假采样。(正负样本不平衡是通过最后loss的OHEM完成)
#    bbox_sampler = PseudoSampler(**sampler_cfg.params)
#    sampling_result = bbox_sampler.sample(*assign_result, anchors, gt_bboxes)
#    pos_inds, neg_inds = sampling_result  # (k,), (j,) 表示正/负样本的位置号，比如[7236, 7249, 8103], [0,1,2...8104..]
#    
#    # 3. 初始化target    
#    bbox_targets = torch.zeros_like(anchors)
#    bbox_weights = torch.zeros_like(anchors)
#    labels = anchors.new_zeros(len(anchors), dtype=torch.long)       # 借用anchors的device
#    label_weights = anchors.new_zeros(len(anchors), dtype=torch.long)# 借用anchors的device
#    # 4. 把正样本 bbox坐标转换成delta坐标并填入
#    pos_bboxes = anchors[pos_inds]                   # (k,4)获得正样本bbox
#    
#    pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1 # 表示正样本对应的label也就是gt_bbox是第0个还是第1个(已经减1，就从1-n变成0-n-1)
#    pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]  # (k,4)获得正样本bbox对应的gt bbox坐标
#    
#    pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means, target_stds)  
#    bbox_targets[pos_inds] = pos_bbox_targets
#    bbox_weights[pos_inds] = 1
#    # 5. 把正样本labels填入
#    labels[pos_inds] = gt_labels[pos_assigned_gt_inds] # 获得正样本对应label
#    label_weights[pos_inds] = 1  # 这里设置正负样本权重都为=1， 如果有需要可以提高正样本权重
#    label_weights[neg_inds] = 1
#    
#    return bbox_targets, bbox_weights, labels, label_weights, pos_inds, neg_inds


# %% anchor free - FCOS的target生成程序
    
def get_centerness_target(pos_bbox_targets):
    """centerness可以理解为中心度，用来表征该点距离bbox中心点的程度，该值越接近1，则越接近bbox中心点。
    而越接近0则越远离中心点。因此采用c = sqrt((min(l,r)/max(l,r) * min(t,b)/max(t,b))来表示，相当于
    用l/r来评估偏离度，且分别评估了水平方向和竖直方向的偏离度。
    args: pos_bbox_targets (k, 4) 代表(l,r,t,b)
    return: target(k,)
    """
    lr = pos_bbox_targets[:, [0, 2]]
    tb = pos_bbox_targets[:, [1, 3]]
    centerness_target = (lr.min(dim=1)[0] / lr.max(dim=1)[0]) * \
                        (tb.min(dim=1)[0] / tb.max(dim=1)[0])
    return torch.sqrt(centerness_target)  # (k,)
    

def get_points(featmap_sizes, strides, device):
    """计算一张图片所有特征子图的网格中心点坐标集合
    args:
        featmap_sizes: (n_level, )(h, w)
        strides: 步长
        device
    returns:
        points: (n_level,)(k, 2) 表示每层的k个中心点
    """
    all_points = []
    for i, featmap_size in enumerate(featmap_sizes):
        h, w = featmap_size
        x = torch.arange(0, w * strides[i], strides[i], device=device)
        y = torch.arange(0, h * strides[i], strides[i], device=device)
        xx, yy = torch.meshgrid(x, y)
        # 先堆叠成(k, 2)然后平移半个步长到网格中心点
        points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1) + strides[i] // 2
        all_points.append(points)
    return all_points   # (5,)(k,2)


    
def get_point_target(points, regress_ranges, gt_bboxes_list, gt_labels_list, num_level_points):
    """计算target
    args:
        points: (5,)(n,2)，单张图的每个特征图的point集合
        regress_ranges: ((-1,64),(64,128),(..),(..),(..))，由于对每张图都一样，所以regress_rnage不需要分batch
        gt_bboxes_list: (b,)(m, 4), b为batch size
        gt_labels_list: (b,)(m,), b为batch size
        num_level_points: (5,) 表示每层特征图上的points个数
    return
        all_bbox_targets
        all_labels
    """
    # 堆叠point
    points = torch.cat(points)  # (k,2)
    # 堆叠regress_ranges：让每个point对应一个regress range
    repeat_regress_ranges = []
    for i, ranges in enumerate(regress_ranges):
        repeat_regress_ranges.append(points.new_tensor(ranges).repeat(num_level_points[i], 1))
    regress_ranges = torch.cat(repeat_regress_ranges, dim=0)  # (k, 2)代表一张图的，跟一张图的points一一对应
    # 计算每张图的target
    num_imgs = len(gt_bboxes_list)
    bbox_targets = []
    labels = []
    for i in range(num_imgs):
        bbox_target, label = point_match_target(
                points, regress_ranges, gt_bboxes_list[i], gt_labels_list[i])
        bbox_targets.append(bbox_targets)  # (b,)(k,4)
        labels.append(labels)             # (b,)(k,)
    # 变换格式：从外层batch，内层(k)变换为外层按照特征分，内层按照batch分
    all_labels = []
    all_bbox_targets = []
    for num in num_level_points:
        stacked_labels = torch.stack([label[:num] for label in labels], dim=0)  # (b,)(n,) ->(m,)
        stacked_bbox_targets = torch.stack([bbox_target[:num] for bbox_target in bbox_targets], dim=0) # (b,)(n,4) ->(m,4)
         
        all_labels.append(stacked_labels)  # (5,) (m,)
        all_bbox_targets.append(stacked_bbox_targets) # (5,) (m,4)
    
    return all_bbox_targets, all_labels  # (5,)(m,4)  and (5,)(m,)
    
    
    
def point_match_target(points, regress_ranges, gt_bboxes, gt_labels):
    """单张图的target计算
    1. 如果point在某一gt bbox内，则该点为正样本，否则为负样本
    2. 如果point对应最大l/r/t/b大于回归值，则该点不适合在该特征图，取为负样本
    args:
        points: (k, 2)
        regress_ranges: (k, 2)
        gt_bboxes: (m, 4)
        gt_labels: (m, )
    """
    num_points = points.shape[0]
    num_gts = gt_labels.shape[0]
    # 计算每个gt bbox的面积: 用于后边取最小面积
    areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1) # (m,)

    # 计算每个point与每个gt bbox的对应样本的l,r,t,b
    left, right, top, bottom = bbox2lrtb(points, gt_bboxes)
    # 得到每个point针对每个gt bbox的target，然后再筛选
    bbox_targets = torch.stack([left,right,top,bottom], dim=-1) # (k, m, 4)
    
    # 1. 判断point是否在gt bbox中(l/r/b/t > 0)
    inside_gt_bbox_mask = bbox_targets.min(dim=-1)[0] > 0  # 用l/r/t/b做判断：取l,r,t,b中最小值>0，则说明l,r,t,b4个值都大于0，必在bbox内
    
    # 2. 判断point的最大回归值是在哪个回归范围，从而判断该点应该在哪张特征图(l/r/b/t < regress_range)
    regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)  # 变换为(k,m,2)便于跟(k,m)做运算
    max_regress_distance = bbox_targets.max(dim=-1)[0]     #(k, m) 
    inside_regress_range_mask = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                    max_regress_distance <= regress_ranges[..., 1])   # 取l,r,t,b中最大值检查是否在regress range内
    # 3. 基于前2步筛除area，同时如果一个point对应了多个gt，取面积更小的gt bbox作为该point对应的bbox
    areas = areas.repeat(num_points, 1)    # (k, m)
    areas[inside_gt_bbox_mask == 0] = 1e+8       # 剔除gt外边的点
    areas[inside_regress_range_mask == 0] = 1e+8  # 剔除回归范围外的点
    min_areas, min_area_inds = areas.min(dim=1)    # 为每个point在m个gt中找到面积最小对应的gt bbox
    
    # 为每一个point指定一个label
    labels = gt_labels[min_area_inds]   # 取最小面积对应的那个bbox的label(都是作为正样本取标签，1-20)
    labels[min_areas == 1e+8] = 0       # (k,)包含了正样本和负样本：如果该最小面积不在gt bbox，也不在回归范围，则属于负样本取标签=0
                                        # 注意：labels和bbox_tagets都包含了所有的正样本和负样本，没有去掉任何样本。
    # 取最小面积ind对应的bbox坐标
    # 注意这里筛选bbox_target的用法很容易忽略出错，是对第0,第1维度分别用列表筛选，得到的是两个列表交汇的位置的坐标，所以降维了。
    bbox_targets = bbox_targets[range(num_points), min_area_inds]   # (k,m,4)-> (k,4)
    return bbox_targets, labels  # (k,4) , (k,)
     


if __name__ == "__main__":
    # 单图points
    points = get_points((300, 300), 10)
    regress_ranges = ((30, 30))
    gt_bboxes = torch.tensor([[0,0,40,40], [100,100, 170, 170]])
    gt_labels = torch.tensor([1, 1])
    bbox_targets, labels = get_point_target(points, regress_ranges, gt_bboxes, gt_labels)        
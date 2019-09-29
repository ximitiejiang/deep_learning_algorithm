#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 08:52:11 2019

@author: ubuntu
"""
import numpy as np
from utils.ious import calc_ious_np

def eval_map(det_results,
             gt_bboxes,
             gt_labels,
             scale_ranges,
             iou_thr=0.5,
             dataset=None,
             print_summary=True):
    """用于对数据集进行均值平均精度(mAP)的评估
    通过设置iou_thr的阈值来评估在预测结果大于某个iou阈值情况下的所有类的平均精度
    最常见的是评估iou_thr=0.5的平均精度，也就是iou>0.5为预测正确，否则为预测错误(也是voc数据集的评测标准)
    更严苛的是评估iou(0.05-0.95)之间的平均精度(比如coco数据集的评测标准)
    args:
        det_results: (n_imgs, )(n_classes, )(k, 5)，是按照img数量，嵌套类别数量，嵌套每个类别的box预测(4列坐标和1列置信度，类别顺序就代表标签)
        gt_bboxes: (n_imgs, )(k, 4)
        gt_labels: (n_imgs, )(k, )
        iou_thr: 交并比的阈值
        dataset: 数据集的名字
        print_summary: 是否打印结果
    """
    
    num_classes = len(det_results[0])
    num_imgs = len(det_results)
    
    eval_results = []
    # 计算每一类的ap
    for i in range(num_classes):
        # 抽取单个类的数据  (n_imgs)(m, 5), (n_imgs)(m, 4) 
        det_cls, gt_bboxes_cls = get_one_class_results(i, det_results, gt_bboxes, gt_labels)
        # 计算单图单类的统计结果
        tpfp = []
        for j in range(num_imgs):
            tpfp.append(tpfp_default(det_cls[j], gt_bboxes_cls[j], iou_thr))
        tp, fp = tuple(zip(*tpfp))  # tp格式(n_imgs,)(1,k),k表示该图有k个预测det
        # 计算总共的gt bboxes个数
        num_gts = 0
        for j, bbox in enumerate(gt_bboxes_cls):
            num_gts += bbox.shape[0]
        # 对该类的所有数据进行堆叠排序
        det_cls = np.concatenate(det_cls, axis=0)    # (k, 5)
        num_dets = det_cls.shape[0]
        sort_inds = np.argsort(det_cls[:, -1])[::-1]  # 按score从大到小排序
        tp = tp.concatenate(tp, dim=1)[:, sort_inds]  # (1,k)按score排序  n张图的所有det的标记共1766个det, 为tp则标记为1， 否则为0
        fp = np.concatenate(fp, dim=1)[:, sort_inds]  # (1,k)
        
        # 计算tp,fp的累计数，也就是生成x方向(recall)坐标和y方向(precision)坐标
        tp_cum = np.cumsum(tp, axis=1)    # (1,k)
        fp_cum = np.cumsum(fp, axis=1)    # (1,k)
        recalls = tp_cum / num_gts                 # 召回率：tp/(tp+fn) 为真阳性/所有gt个数，可以看到分母是不变的，说明R是逐渐增加
        precisions = tp_cum / (tp_cum + fp_cum)    # 精确率: tp/(tp+fp) 为真阳性/预测个数，可以看到分母是逐步加1，说明P是逐渐减小
        
        recalls = recalls.reshape(-1)
        precisions = precisions.reshape(-1)
        mode = 'area'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append(dict(
                num_gts = num_gts,
                num_dets = num_dets,
                recall = recalls,
                precision = precisions,
                ap = ap))
    # 汇总
    aps = []
    for result in eval_results:
        if result['num_gts'] > 0: # 如果有
            aps.append(result['ap'])
    mean_ap = np.array(aps).mean().item() if aps else 0.0
    # 打印
    if print_summary:
        print_map_summary(mean_ap, eval_result, dataset)
    
    return mean_ap, eval_results
    
        


def get_one_class_results(class_id, det_results, gt_bboxes, gt_labels):
    """从总的预测数据中提取对某一类的所有图片数据"""
    # 提取单类的det_result
    det_results_cls = [det[class_id] for det in det_results]
    # 提取单类的gt_bbox, gt_label
    gt_bboxes_cls = []
    for i in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[i]
        cls_inds = (gt_labels[i] == class_id + 1)  # label是从1-20所以class_id需要+1才能与label匹配
        gt_bbox_cls = gt_bbox[cls_inds, :] if gt_bbox.shape[0] > 0 else gt_bbox
        gt_bboxes_cls.append(gt_bbox_cls)

    return det_results_cls, gt_bboxes_cls  # (n_imgs,)(k,5),  (n_imgs,)(k, 4)


def tpfp_default(det_bboxes, gt_bboxes, iou_thr):
    """统计单张图的两组bboxes下，真阳性样本和假阳性样本的数量
    tpfp代表true positive, false positive, 表示对真阳性和假阳性的统计
    args:
        det_bboxes: (m,5), 最后一列是score
        gt_bboxes: (m,4)
    """
    
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    # 初始化tp, fp，
    tp = np.zeros((1, num_dets), dtype=np.float32)
    fp = np.zeros((1, num_dets), dtype=np.float32)
    # 如果没有gt却有det，则所有det都是fp
    if num_gts == 0:
        fp[...] = 1
        return tp, fp
    # 如果有gt
    else:
        ious = calc_ious_np(det_bboxes[:,:4], gt_bboxes) # (m,n) m个预测
        ious_max = ious.max(axis=1)        # 取m个预测中每个预测的最大匹配iou值
#        ious_argmax = ious.argmax(axis=1)  # 取m个预测中每个预测的最大匹配iou的位置
        sort_inds = np.argsort(det_bboxes[:, -1])[::-1]  # 按置信度从大到小对预测score排序
        for i in sort_inds:  # 按置信度大到小取iou出来判断
            if ious_max[i] >= iou_thr:     # iou大于等于阈值则为真阳性
#                matched = ious_argmax[i] 
                tp[:, i] = 1  
            else:                          # iou小于阈值则为假阳性
                fp[:, i] = 1
        return tp, fp
                

def average_precision(recalls, precisions, mode):
    """计算单个类的平均精度
    args:
        recalls: (n_dets,) 代表随着预测正样本增加(分子)，recall的提高
        precisions: (n_dets, ) 代表随着预测样本个数增加(分母)，precisions的下降
        mode: 'area'表示在precision-recall曲线之下计算面积来作为平均精度
              '11points'表示在[0,0.1,...1]这11个点下计算平均精度
    """
    recalls = recalls.reshape(1, -1)
    precisions = precisions.reshape(1, -1)
    recall_x = np.concatenate([np.zeros((1,1)), recalls, np.ones((1,1))])     # 在前面加0，后边加1，组成x坐标 
    precision_y = np.concatenate([np.zeros((1,1)), recalls, np.zeros((1,1))]) # 在前面加0，后边加0
    
                
def print_map_summary(mean_ap, results, dataset=None):   
    """用于打印mAP的结果，包括计算的recall, precision, ap, map
    打印效果如下：
    +-------------+------+-------+--------+-----------+-------+
    | class       | gts  | dets  | recall | precision | ap    |
    +-------------+------+-------+--------+-----------+-------+
    | aeroplane   | 285  | 1799  | 0.800  | 0.127     | 0.728 |
    | bicycle     | 337  | 1766  | 0.825  | 0.157     | 0.704 |
    | bird        | 459  | 2138  | 0.765  | 0.164     | 0.615 |
    | boat        | 263  | 3495  | 0.802  | 0.060     | 0.540 |
    | bottle      | 469  | 7476  | 0.646  | 0.041     | 0.322 |
    | bus         | 213  | 1860  | 0.878  | 0.101     | 0.756 |
    | car         | 1201 | 7794  | 0.907  | 0.140     | 0.798 |
    | cat         | 358  | 1821  | 0.925  | 0.182     | 0.812 |
    | chair       | 756  | 6875  | 0.684  | 0.075     | 0.378 |
    | cow         | 244  | 1675  | 0.893  | 0.130     | 0.609 |
    | diningtable | 206  | 2697  | 0.845  | 0.065     | 0.609 |
    | dog         | 489  | 2179  | 0.896  | 0.201     | 0.729 |
    | horse       | 348  | 1369  | 0.871  | 0.221     | 0.764 |
    | motorbike   | 325  | 1686  | 0.837  | 0.161     | 0.700 |
    | person      | 4528 | 27640 | 0.867  | 0.142     | 0.695 |
    | pottedplant | 480  | 10155 | 0.762  | 0.036     | 0.334 |
    | sheep       | 242  | 1588  | 0.839  | 0.128     | 0.599 |
    | sofa        | 239  | 2270  | 0.895  | 0.094     | 0.621 |
    | train       | 282  | 1606  | 0.879  | 0.154     | 0.761 |
    | tvmonitor   | 308  | 2705  | 0.802  | 0.091     | 0.602 |
    +-------------+------+-------+--------+-----------+-------+
    | mAP         |      |       |        |           | 0.634 |
    +-------------+------+-------+--------+-----------+-------+
    """             
    pass
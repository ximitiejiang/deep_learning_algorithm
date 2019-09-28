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
    
    for i in range(num_classes):
        # 抽取单个类的数据
        det_cls, gt_bboxes_cls = get_one_class_results(i, det_results, gt_bboxes, gt_labels)
        # 计算单图单类的统计结果
        for j in range(num_imgs):
            tpfp = tpfp_default(det_cls[j], gt_bboxes_cls[j], iou_thr)
            tp, fp = tpfp
        #
        
        


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
    """统计
    tpfp代表true positive, false positive, 表示对真阳性和假阳性的统计"""
    
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    
    tp = np.zeros((1, num_dets), dtype=np.float32)
    fp = np.zeros((1, num_dets), dtype=np.float32)
    # 如果没有gt却有det，则所有det都是fp
    if num_gts == 0:
        fp[...] = 1
        return tp, fp
    else:
        ious = calc_ious_np()

         
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:29:43 2019

@author: ubuntu
"""
"""检测器在整个coco数据集上的评估方法？
1. coco数据集的评估从官网上看需要评估12个参数如下：
参考：https://www.jianshu.com/p/d7a06a720a2b (非常详细介绍了两大数据集检测竞赛的评价方法包括源码)
参考：https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py (coco官网的eval代码)
注意：coco并不区分AP和mAP，以及AR和mAR，一般都是指mAP(均值平均精度)即所有类别的平均精度,并且AP是coco最核心的一个指标，AP高的取胜。
所以coco最常用mAP@IoU=0.5:0.95

(AP)Average Precision
    AP IoU=0.5:0.95             # 在iou = [0.5,0.95,0.05]范围内的平均AP
    AP IoU=0.5                  # 在iou = 0.5的平均AP(这也是voc的要求)
    AP IoU=0.75                 # 在iou = 0.75的平均AP(更严格的要求)
(AP)Average Precision across scales:
    AP IoU=0.5:0.95 small       # 小目标的检测AP(area< 32*32), 约41%
    AP IoU=0.5:0.95 medium      # 中目标的检测AP(32*32<area<96*96), 约34%
    AP IoU=0.5:0.95 large       # 大目标的检测AP(area> 96*96), 约24%
    其中面积通过mask像素数量计算
(AR)Average Recall
    AR IoU=0.5:0.95 max=1        # 一张图片图片给出最多1个预测
    AR IoU=0.5:0.95 max=10       # 一张图片图片给出最多10个预测
    AR IoU=0.5:0.95 max=100      # 一张图片图片给出最多100个预测
(AR)Average Recall across scales:
    AR IoU=0.5:0.95 small        # 小目标的召回率
    AR IoU=0.5:0.95 medium       # 中目标的召回率
    AR IoU=0.5:0.95 large        # 大目标的召回率

2. voc的评价方法虽然也是box AP为主，但计算方法稍有不同：
voc不像coco是把从0.5:0.95的10个阈值算出来的AP进行平均，高iou阈值的AP肯定导致精度下降，因此voc方式算出来mAP远比coco方式高
所以voc最常用mAP@IoU=0.5

"""
from six.moves import cPickle as pickle
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

#import _pickle as pickle
"""注意cPickle, Pickle, six.moves的区别：
1. cPickle是c代码写成，Pickle是python写成，相比之下cPickle更快
2. cPickle只在python2中存在，python3中换成_pickle了
3. six这个包是用来兼容python2/python3的，这应该是six的由来(是2与3的公倍数)
   six包里边集成了有冲突的一些包，所以可以从里边导入cPickle这个在python3已经取消的包
"""



def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def _recalls(all_ious, proposal_nums, thrs):
    """计算召回率recalls: 
    计算过程：先计算所有图片中gt_bbox跟proposals的ious,
    在1种topk(比如top100个proposal)情况下，找到10个iou阀值下所有类别recall的平均值
    Args:
        all_ious(array): (k,m,n) k张图，m个gt, n个proposals
        proposal_nums(array): [100,300,1000]代表取前100/300/1000个proposals 
        thrs(array): [0.50,0.55,0.60,...0.95]共10个iou阈值等级
    Return:
        recalls(): (3,10)
    """
    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious]) 

    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32) # (3,10)
    for k, proposal_num in enumerate(proposal_nums): # 外层：3种topk
        tmp_ious = np.zeros(0)
        for i in range(img_num): # 中层：n张图片
            ious = all_ious[i][:, :proposal_num].copy() # 取出第i张图片的ious的前num个(但iou每行都是从大到小排序的吗？)
            gt_ious = np.zeros((ious.shape[0]))
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):  # 内层：每张图片的j个gt bboxes
                gt_max_overlaps = ious.argmax(axis=1) # 每个gt在该行对应最大iou(第x列)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]# 取出每行最大iou值
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size)) # (3,10)
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num) # recall计算

    return recalls


def set_recall_param(proposal_nums, iou_thrs):
    """Check proposal_nums and iou_thrs and set correct format.
    """
    if isinstance(proposal_nums, list):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums

    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, list):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs

    return _proposal_nums, _iou_thrs


def eval_recalls(gts,
                 proposals,
                 proposal_nums=None,
                 iou_thrs=None,
                 print_summary=True):
    """Calculate recalls.

    Args:
        gts(list or ndarray): a list of arrays of shape (n, 4) 总计5000个，代表每张图有一组gt bboxes
        proposals(list or ndarray): a list of arrays of shape (k, 4) or (k, 5)
        proposal_nums(int or list of int or ndarray): top N proposals
        thrs(float or list or ndarray): iou thresholds

    Returns:
        ndarray: recalls of different ious and proposal nums
    """

    img_num = len(gts)
    assert img_num == len(proposals)
    # 这里proposals是5000张图的所有预测bboxes汇总
    # proposal_nums是看提取前top100,top300, top1000 ???
    proposal_nums, iou_thrs = set_recall_param(proposal_nums, iou_thrs)

    all_ious = []
    for i in range(img_num):
        
        if isinstance(proposals[i], list):  # 新增一个分支
            new_prop = []
            for j in range(len(proposals[i])):
                if proposals[i][j].size != 0:
                    new_prop.append(proposals[i][j])
            if len(new_prop) != 0:
                img_proposal = np.concatenate(new_prop, axis=0)
            else:
                img_proposal = np.zeros((0,4),dtype=np.float32)
        
        elif proposals[i].ndim == 2 and proposals[i].shape[1] == 5: # 如果一张图片的proposal是(n,5)形式
            scores = proposals[i][:, 4]               # 取出最后一列scores置信度 
            sort_idx = np.argsort(scores)[::-1]       # scores排序，从大到小
            img_proposal = proposals[i][sort_idx, :]  # 取出按score排序出的proposals
        else:
            img_proposal = proposals[i]
    
        prop_num = min(img_proposal.shape[0], proposal_nums[-1]) # 如果proposal个数少于则取少的那个值
        if gts[i] is None or gts[i].shape[0] == 0:  # 如果没有gt
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            ious = bbox_overlaps(gts[i], img_proposal[:prop_num, :4]) # 计算gt与proposals的ious
                                                                      # 也就要求proposal[i]需要是一张图片所有class的总和
        all_ious.append(ious)
    all_ious = np.array(all_ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)  # (3,10)
    if print_summary:
        print_recall_summary(recalls, proposal_nums, iou_thrs)
    return recalls


def print_recall_summary(recalls,
                         proposal_nums,
                         iou_thrs,
                         row_idxs=None,
                         col_idxs=None):
    """Print recalls in a table.

    Args:
        recalls(ndarray): calculated from `bbox_recalls`
        proposal_nums(ndarray or list): top N proposals
        iou_thrs(ndarray or list): iou thresholds
        row_idxs(ndarray): which rows(proposal nums) to print
        col_idxs(ndarray): which cols(iou thresholds) to print
    """
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    iou_thrs = np.array(iou_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header=[]
    for i in range(len(iou_thrs)):
        row_header.append(round(iou_thrs[i],2))
    row_header = [''] + row_header
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [
            '{:.3f}'.format(val)
            for val in recalls[row_idxs[i], col_idxs].tolist()
        ]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print(table.table)

def evaluation(result_file_path, coco_obj, eval_types = ['bbox']):
    """基于已经生成好的pkl或json模型预测结果文件，进行相关操作:
    Args:
        result_file_path(str): .pkl or .pkl.json file
        coco_obj(obj): coco object belong to COCO class
        eval_types(list): ['proposal_fast', 'bbox', 'proposal']
    Return:
        None
    假定result.pkl已经获得则可按如下进行评估，但实际的test forward()计算过程如下
    在detector的forward_test()函数中, 内部调用simple_test()
        - 从backbone/neck获得x: 从img(1,3,800, 1216)到x[(1,256,200,304),(1,256,100,152),(1,256,50,76),(1,256,25,38),(1,256,13,39)]
        - 再从RPN head调用simple_test_rpn()
            获得rpn head的输出rpn_outs = rpn_head(x), 输出结构2个元素，[cls_scores, bbox_preds], 每个都是5层
            获得proposal_list = rpn_head.get_bboxes(), 输出结构1个元素，[(2000,5)]
        - 再从Bbox head调用simple_test_bboxes()
            获得rois = bbox2rois(proposals), 输出结构(2000,5)
            获得roi_feats = bbox_roi_extractor(x, rois)，输出结构(2000,256,7,7)
            获得bbox head的输出cls_score/bbox_pred = bbox_head(roi_feats)，输出结构(2000,81)和(2000,324)
            获得det_bboxes, det_labels = bbox_head.get_det_bboxes()，输出结构(100, 5)和(100,)
         - 最后调用bbox2result()从det_bboxes, det_labels中筛选出results, 结构为[class1, class2, ...]，
           每个class为(n,5)数据，代表预测到的该类的bbox个数和置信度
         - 对于整个数据集的single_test，一张图片会对应一个outputs，所以：
           最终outputs结构：外层list长5000，中层list分类长80，内层array bbox尺寸(n,5)    
         - 数据保存：
             如果是为了做proposal_fast评测(评估AR)，按原样保存outputs变量成pkl文件即可
             如果是为了做proposal/bbox评测(评测AP)，则需要修改outputs变量成json支持的dict格式保存(json不支持array)
                 格式修改过程参考results2json()
    """
        # 创建coco对象: 基于val数据集
    coco = coco_obj
    if eval_types:# 假定数据文件已经处理保存好了(pkl or json)
        print('Starting evaluate {}'.format(' and '.join(eval_types)))
        if eval_types == ['proposal_fast']:  # 如果是快速方案验证，则提取对应pkl文件(直接outputs保存，外层list5000,中层list80, 内层array(n,5))
            result_file = result_file_path
            with open(result_file_path, 'rb') as f:
                results = pickle.load(f)   

            # 从coco数据集中提取ann info
            gt_bboxes = []
            img_ids = coco.getImgIds()
            for i in range(len(img_ids)):
                ann_ids = coco.getAnnIds(imgIds = img_ids[i])  # 一个img_id对应多个ann_inds，这多个ann_inds会组成一个ann_info
                ann_info = coco.loadAnns(ann_ids)
                if len(ann_info) == 0: # 如果是空
                    gt_bboxes.append(np.zeros((0,4)))
                    continue
                # 提取每个img的ann中的信息
                bboxes=[]
                for ann in ann_info:
                    if ann.get('ignore', False) or ann['iscrowd']:
                        continue
                    x1,y1,w,h = ann['bbox']
                    bboxes.append([x1,y1,x1+w-1,y1+h-1])
                bboxes = np.array(bboxes, dtype = np.float32)  # 这里要转成ndarray做什么？
                if bboxes.shape[0] == 0:
                    bboxes = np.zeros((0,4))
                gt_bboxes.append(bboxes)
            # 计算recalls    
            max_dets = np.array([100, 300, 1000])   # 代表average recall的前100
            iou_thrs = np.arange(0.5, 0.96, 0.05)   # [0.5,0.55,0.60,...0.95]
            
            # 似乎这个proposal_fast的eval_recalls()只支持proposal格式[array1,...array5000],每个array(m,5)
            # 不支持propsal已经转换成按类分类的方式。
            
            recalls = eval_recalls(gt_bboxes, 
                                   results, 
                                   max_dets, 
                                   iou_thrs, 
                                   print_summary=True)
            avg_recall = recalls.mean(axis=1)       # 计算AR(average recall) (3,10)
            # 显示recall
            for i, num in enumerate(max_dets):
                print('AR@{}\t= {:.4f}'.format(num, avg_recall[i]))
        
        else:  # 如果是coco api的方案验证，则提取对应json文件(转换格式，外层list208750，内层dict('img_id','bbox','score','category'))
            """cocoEval需参考：https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
            注意：
            1. coco.loadRes()为加载预测结果文件到coco对象，该预测结果格式必须为json, [dict1,..dictn]每个bbox一个dict
               每个dict(('img_id','bbox','score','category')
            2. 把模型输出outputs转换到以上coco认可的格式，需要通过results2json进行转换，可直接从pkl文件进行转换
            """
            result_file = result_file_path
            with open(result_file, 'r') as f:
                results = json.load(f) 
            coco_dets = coco.loadRes(result_file) # coco api要求的结果形式：json, 外层list，内层dict
            img_ids = coco.getImgIds()
            # 定义iou_type: 在coco中iou_type = ['bbox','segm', 'keypoints']三种选择，物体检测需要选bbox
            # 区别eval_types: proposals, bbox
            for res_type in eval_types:
                if res_type == 'proposal':
                    iou_type = 'bbox'
                else:
                    iou_type = res_type
                cocoEval = COCOeval(coco, coco_dets, iou_type)
                cocoEval.params.imgIds = img_ids
                if res_type == 'proposal':
                    cocoEval.params.useCats = 0
                    cocoEval.params.maxDets = list(max_dets)
                
                cocoEval.evaluate()     # 对每一张图片分别评估，耗时较长(1)
                cocoEval.accumulate()   # 
                cocoEval.summarize()    # 结果中AP@IoU=0.5:0.95为0.364，跟faster rcnn披露出来的box AP一致



if __name__=='__main__':
    eval_with_pkl_file = True
    if eval_with_pkl_file:
        data_root = 'data/coco/'    # 需要预先把主目录加进sys.path
        ann_file=[data_root + 'annotations/instances_train2017.json',
                  data_root + 'annotations/instances_val2017.json']
        coco = COCO(ann_file[1])  #验证集
        
        eval_types = ['bbox']  # 可选择['proposal_fast']或['bbox', 'proposal']
        result_file_path = 'data/coco/results.pkl.json'  # 可选择pkl文件或者json文件
        
        evaluation(result_file_path, coco, eval_types = eval_types)     

        

            
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:26:28 2019

@author: ubuntu
"""
#
#from utils.visualization import vis_dataset_bbox_area
#from utils.dataset_classes import get_classes
#
#
#cfg_path = '../demo/cfg_detector_ssdvgg16_voc.py'
#vis_dataset_bbox_area(cfg_path)


"""参考 https://github.com/lars76/kmeans-anchor-boxes"""

from tqdm import tqdm
import numpy as np
import torch
from utils.visualization import vis_bbox
from utils.prepare_training import get_config, get_dataset

# %% 重写kmean方法

def load_dataset(cfg_path):
    """加载经过变换之后的数据集，返回实际训练时的img尺寸和bbox尺寸
    """
    cfg = get_config(cfg_path)
    dataset = get_dataset(cfg.trainset, cfg.transform)
    ww = []
    hh = []
#    data = dataset[0]
    for data in tqdm(dataset):
        bbox = data['gt_bboxes']
        w = (bbox[:, 2] - bbox[:, 0]).numpy()
        h = (bbox[:, 3] - bbox[:, 1]).numpy()
        ww.append(w)
        hh.append(h)
    ww = np.concatenate(ww, axis=0) # (m,)     
    hh = np.concatenate(hh, axis=0)
    bboxes = np.concatenate([ww[:, None], hh[:, None]], axis=1)    
    return bboxes  # (m, 2)


def bbox_kmeans(bboxes, k, method=np.median):
    """计算数据集中所有bbox的聚类，聚类的特征是"""
    num_bboxes = bboxes.shape[0]
    # 随机提取k个作为种子聚类点
    k_clusters = bboxes[np.random.choice(num_bboxes, k)]   # (k, 2)
    # 初始化距离
    distances = np.empty((num_bboxes, k))  # (m,k)
    last_clusters = np.zeros((num_bboxes, ))  # (m,)
    n_iter = 0
    while True:
        # 计算每个bbox跟种子聚类点的距离
        for row in range(num_bboxes):
            distances[row] = 1 - area_iou(bboxes[row], k_clusters)
        # 指定每个bbox所属的聚类点：相当于完成一次聚类
        nearest_clusters = np.argmin(distances, axis=1)
        # 判断该次聚类后是否已ok，如果任何一个bbox的聚类结果都没有变化(.all()都为1)，说明聚类稳定了，退出
        if (nearest_clusters == last_clusters).all():
            break
        else:  # 如果聚类没有稳定，则用聚类结果的均值更新种子点，
            print('iter {} \tavg_distance: {}'.format(n_iter, distances.sum() / num_bboxes))
            for ki in range(k):
                k_clusters[ki] = method(bboxes[nearest_clusters == ki], axis=0)  # 基于聚类结果更新k个种子点
            last_clusters = nearest_clusters  # 
        n_iter += 1
    return k_clusters # (k, 2)


def area_iou(box, clusters):
    """面积交并比：计算两组bbox的面积尺寸的交并比(不是位置的交并比)
    args:
        box:  (2,)某一个box的w,h
        clusters: (k, 2)某一组种子聚合点
    return:
        iou (k,)
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    area_intersection = x * y
    area_box = box[0] * box[1]
    area_clusters = clusters[:, 0] * clusters[:, 1]  # (k,)
    # TODO: 需要预防分母为0或者nan的情况(在widerface中可能出现)
    iou = area_intersection / (area_box + area_clusters - area_intersection) # (k,)
    
    return iou # (k,)


def avg_area_iou(bboxes, clusters):
    """计算聚类的k个bbox跟整个数据集的bboxes的平均面积iou: 
    注意每个bbox跟聚类的k个bbox都可以有k个iou，其中iou最大的就是这个bbox对应的iou，因为所聚类的bbox会密布整张图，iou最大那个就会匹配到。
    args:
        bbox: (m, 2), 两列为w,h
        clusters: (k, 2), 两列为w,h
    return:
        avg_iou (item)
    """
    avg_iou = np.mean([np.max(area_iou(bboxes[i], clusters)) for i in range(bboxes.shape[0])])
    return avg_iou


def get_voc_anchor_params(cfg_path, k, img_size=None):
    """获取voc数据集的anchor参数, 如果输入了图片尺寸，则得到的是实际anchor尺寸。
    注意：由于数据集的数据会做预处理，实际的图片尺寸和实际的bbox尺寸都是需要先经过transform才能得到
    所以最好基于预处理之后的尺寸进行聚类分析。
    
    args:
        xml_path: str
        k: int
        img_size: (w, h)
    """
    bbox_data = load_dataset(cfg_path)
    # 计算从小到大的clusters
    k_clusters = bbox_kmeans(bbox_data, k)
    inds = np.argsort(k_clusters[:, 0] * k_clusters[:, 1])
    k_clusters = k_clusters[inds]  
    
    # 计算平均面积iou指标
    avg_ious_result = avg_area_iou(bbox_data, k_clusters)
    ratios = (k_clusters[:, 0] / k_clusters[:, 1]).tolist()     # w/h
    print('avg ious for %d clusters: %f'%(k, avg_ious_result))
    print('kmean anchors w and h: \n {}'.format(k_clusters))
    print('kmean anchors ratios(w/h): \n {}'.format(ratios))
    
    vis_bbox(k_clusters)
    return k_clusters, ratios, avg_ious_result
    

def get_voc_avg_area_iou(anchors, cfg_path):
    """在已有anchors的条件下，根据计算avg_area_iou来评估该anchors跟数据集的匹配程度
    args:
        anchors: (k, 2) or (k, 4), 如果是(k,2)则为w,h, 如果是(k, 4)则为xmin,ymin,xmax,ymax
    """
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.numpy()     
    if anchors.shape[1] == 4:
        w = anchors[:, 2] - anchors[:, 0]   # (k,)   
        h = anchors[:, 3] - anchors[:, 1]   # (k,)
        anchors = np.concatenate([w[:,None], h[:, None]], axis=1)

    bbox_data = load_dataset(cfg_path)        
    avg_iou_result = avg_area_iou(bbox_data, anchors)
    print('avg iou: %f'%avg_iou_result)
    return avg_iou_result
    


if __name__ =='__main__':
    task = 'voc'
    
    if task == 'voc':    
        cfg_path = '../demo/cfg_detector_ssdvgg16_voc.py'
        
    if task == 'widerface':
        cfg_path = '../demo/cfg_detector_ssdvgg16_widerface.py'
    
 
    k_clusters, *_ = get_voc_anchor_params(cfg_path, k=9)  # k=9(0.672), k=12

#    get_voc_avg_area_iou(k_clusters, cfg_path)
    
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


"""参考 kmeans-anchor-boxes"""

import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np

ANNOTATIONS_PATH = "/home/ubuntu/MyDatasets/voc/VOCdevkit/VOC2007/Annotations"
CLUSTERS = 5

def load_dataset(path):
	dataset = []
	for xml_file in tqdm(glob.glob("{}/*xml".format(path))):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			xmin = int(obj.findtext("bndbox/xmin")) / width
			ymin = int(obj.findtext("bndbox/ymin")) / height
			xmax = int(obj.findtext("bndbox/xmax")) / width
			ymax = int(obj.findtext("bndbox/ymax")) / height

			dataset.append([xmax - xmin, ymax - ymin])  # 这里得到的是相对于图片w,h的百分百(0-1之间)

	return np.array(dataset)

# 获得所有w,h

def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]  # 从rows里边随机选择k个作为种子点，相当于从boxes里边随机抽取5行

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters) # 计算所有点跟种子点的距离，这里距离公式为d = 1 - iou，也就是覆盖度越大，距离越近，覆盖度越小，距离越远。
                                                           # 这里distance记录的就是这组clusters跟每个box的重叠度，类似损失函数的概念，distance越大说明重叠度越差
        nearest_clusters = np.argmin(distances, axis=1)    # (m, ) 把所有点分配到不同的种子点去，相当于基于这k个种子点完成一轮聚类。 

        if (last_clusters == nearest_clusters).all():      # 反复循环，直到所有的点归属于种子点的属性不再变化，则退出循环。
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)   # 更换种子点：从所有点里边获取跟该种子点已聚类的所有点，对这些点求平均，作为新的种子点，
                                                                                   # 新的种子点必然更靠近相关聚类点 
        last_clusters = nearest_clusters

    return clusters


def iou(box, clusters):
    """ 这个交并比计算的目的是计算选出来的clusters跟任意个box的面积之间的关系，返回的iou(5,)就表示这5个clusters相对于box的重合度
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

# %% 重写kmean方法
    
def load_xml(xml_path):
    """加载数据集的xml文件获取所有bbox的信息，返回每个bbox相对于图片尺寸的相对百分比。
    args:
        xml_path: xml所有文件所在文件夹路径
    return
        data: (m, 2)
    """
    data = []
    for xml_file in tqdm(glob.glob("{}/*xml".format(xml_path))):
        tree = ET.parse(xml_file)
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height
            data.append([xmax - xmin, ymax - ymin])  # 这里得到的是相对于图片w,h的百分百(0-1之间)

    return np.array(data)  # (m, 2)


def bbox_kmeans(bboxes, k, method=np.median):
    """计算数据集中所有bbox的聚类，"""
    num_bboxes = bboxes.shape[0]
    # 随机提取k个作为种子聚类点
    clusters = bboxes[np.random.choice(num_bboxes, k)]
    # 初始化距离
    distances = np.empty((num_bboxes, k))
    while True:
        for row in range(num_bboxes):
            distances[row] = 1 - area_iou(bboxes[row], clusters)


def area_iou(box, clusters):
    """面积交并比：计算两组bbox的面积尺寸的交并比(不是位置的交并比)
    args:
        box:  (2,)某一个box的w,h
        clusters: (k, 2)某一组种子聚合点
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    area_intersection = x * y
    area_box = box[0] * box[1]
    area_clusters = clusters[:, 0] * cluster[:, 1]
    
    iou = area_intersection / (area_box + area_clusters - area_intersection) #
    
    return iou # ()


if __name__ =='__main__':
    data = load_dataset(ANNOTATIONS_PATH)
    out = kmeans(data, k=CLUSTERS)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}".format(out))
    
    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))

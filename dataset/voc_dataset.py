#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 18:40:47 2019

@author: ubuntu
"""
import numpy as np
import cv2
import xml.etree.ElementTree as ET

from dataset.base_dataset import BasePytorchDataset

class VOCDataset(BasePytorchDataset):
    """VOC数据集：用于物体分类和物体检测
    2007+2012数据总数16551(5011 + 11540), 可以用2007版做小数据集试验。
    输入：
        root_path
        ann_file
        subset_path
        difficult: 困难bbox(voc中有部分bbox有标记为困难，比如比较密集的bbox，当前手段较难分类和回归)，一般忽略
    """
    
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    
    def __init__(self,
                 root_path=None,
                 ann_file=None,
                 subset_path=None,
                 img_transform=None,
                 label_transform=None,
                 bbox_transform=None,
                 data_type=None,
                 difficult=False):
        # 变换函数
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.bbox_transform = bbox_transform
        
        self.ann_file = ann_file
        self.subset_path = subset_path
        # TODO: 是否取0-19更合适？
        self.class_label_dict = {cat: i+1 for i, cat in enumerate(self.CLASSES)}  # 从1开始(1-20). 
        # 加载图片标注表(只是标注所需文件名，而不是具体标注值)，额外加了一个w,h用于后续参考
        self.img_anns = self.load_annotation_inds(self.ann_file) 
    
    def load_annotation_inds(self, ann_file):
        """从多个标注文件读取标注列表，过程是： 读取ann.txt获得图片名列表 -> 获得img/xml文件名 -> 打开img,打开xml
        """
        img_anns = []
        for i, af in enumerate(ann_file): # 分别读取多个子数据源
            with open(af) as f:
                img_ids = f.readlines()
            for j in range(len(img_ids)):
                img_ids[j] = img_ids[j][:-1]  # 去除最后的\n字符
            # 基于图片id打开annotation文件，获取img/xml文件名
            for img_id in img_ids:
                img_file = self.subset_path[i] + 'JPEGImages/{}.jpg'.format(img_id)
                xml_file = self.subset_path[i] + 'Annotations/{}.xml'.format(img_id)
                # 解析xml文件
                tree = ET.parse(xml_file)
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text) 
                img_anns.append(dict(img_id=img_id, img_file=img_file, xml_file=xml_file, width=width, height=height))
        return img_anns
    
    def parse_ann_info(self, idx):
        """解析一张图片所对应的xml文件，提取相关ann标注信息：bbox, label
        """
        xml_file = self.img_anns[idx]['xml_file']
        tree = ET.parse(xml_file)
        root = tree.getroot()
        bboxes = []
        labels = []
        # 存放difficult=1的困难数据
        bboxes_difficult = []  
        labels_difficult = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.class_label_dict[name]    # label range (1-20)            
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            # 困难数据单独存放
            if difficult:  
                bboxes_difficult.append(bbox)
                labels_difficult.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        # 如果没有bbox数据：则放空
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        # 如果没有困难数据：则放空
        if not bboxes_difficult:
            bboxes_difficult = np.zeros((0, 4))
            labels_difficult = np.zeros((0, ))
        else:
            bboxes_difficult = np.array(bboxes_difficult, ndmin=2) - 1
            labels_difficult = np.array(labels_difficult)
        # 组合数据包为dict
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_difficult=bboxes_difficult.astype(np.float32),
            labels_difficult=labels_difficult.astype(np.int64))
        return ann
        
    def __getitem__(self, idx):
        # 读取图片
        img_info = self.img_anns[idx]
        img_path = img_info['img_file']
        img = cv2.imread(img_path)
        # 读取bbox, label
        ann_dict = self.parse_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        
        # transform
        if self.aug_transform is not None:
            img, gt_bboxes, gt_labels = self.aug_transform(img, gt_bboxes, gt_labels)
        
        img, ori_shape, scale_factor = self.img_transform(img)
        
        gt_bboxes = self.bbox_transform(gt_bboxes)
        
        # 组合数据
        data = 0
        # 如果gt bbox数据缺失，则重新随机一张图片
        while True:
            if len(gt_bboxes) == 0:
                idx = np.random.choice()
                self.__getitem__(idx)
            return data

        
    
    def __len__(self):
        return len(self.img_infos)
    
if __name__ == '__main__':
    data_root_path='/home/ubuntu/MyDatasets/voc/VOCdevkit/'
    params=dict(
                root_path=data_root_path, 
                ann_file=[data_root_path + 'VOC2007/ImageSets/Main/trainval.txt',
                          data_root_path + 'VOC2012/ImageSets/Main/trainval.txt'],
                subset_path=[data_root_path + 'VOC2007/',
                          data_root_path + 'VOC2012/'],
                data_type='train')
    
    dataset = VOCDataset(**params)
    dataset[0]    

    
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
    2007版+2012版数据总数16551(5011 + 11540), 可以用2007版做小数据集试验。
    主要涉及3个文件夹：ImageSets/Main/文件夹中包含.txt文件作为ann_file存放图片名称列表。
                        (一般包含有所有类别的train.txt, val.txt, trainval.txt, test.txt，也包含有每种类别的，所以可以抽取其中几种做少类别二分类或多分类数据集)
                      JPEGImages/文件夹包含.jpg文件，基于图片名称列表进行调用
                      Annotations/文件夹包含.xml文件，基于图片名称列表进行调用
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
                 aug_transform=None,
                 data_type=None,
                 with_difficult=False):
        self.with_difficult = with_difficult
        self.data_type = data_type
        # 变换函数
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.bbox_transform = bbox_transform
        self.aug_transform = aug_transform
        
        self.ann_file = ann_file
        self.subset_path = subset_path
        # TODO: 这里改成了从0-19，是否要保持跟mmdetection一致成1-20？
        self.class_label_dict = {cat: i for i, cat in enumerate(self.CLASSES)}  # 从1开始(1-20). 
        # 加载图片标注表(只是标注所需文件名，而不是具体标注值)，额外加了一个w,h用于后续参考
        self.img_anns = self.load_annotation_inds(self.ann_file) 
        
    
    def load_annotation_inds(self, ann_file):
        """从多个标注文件读取标注列表
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
                # TODO: 这部分解析是否可去除，原本只是为了获得img的w,h,在img_transform里边已经搜集
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
        """虽然ann中有解析出difficult的情况，但这里简化统一没有处理difficult的情况，只对标准数据进行输出。
        """
        # 读取图片
        img_info = self.img_anns[idx]
        img_path = img_info['img_file']
        img = cv2.imread(img_path)
        
        # 读取bbox, label
        ann_dict = self.parse_ann_info(idx)
        gt_bboxes = ann_dict['bboxes']
        gt_labels = ann_dict['labels']

        # aug transform
        if self.aug_transform is not None:
            img, gt_bboxes, gt_labels = self.aug_transform(img, gt_bboxes, gt_labels)
        # basic transform
        if self.img_transform is not None:    
            # img transform
            img, ori_shape, scale_shape, pad_shape, scale_factor, flip = self.img_transform(img)
            # bbox transform: 传入的是scale_shape而不是ori_shape        
            gt_bboxes = self.bbox_transform(gt_bboxes, scale_shape, scale_factor, flip)
            # label transform
            gt_labels = self.label_transform(gt_labels)
        # 组合img_meta
        img_meta = dict(ori_shape = ori_shape,
                        scale_shape = scale_shape,
                        pad_shape = pad_shape,
                        scale_factor = scale_factor,
                        flip = flip)
        # 组合数据: 注意img_meta数据无法堆叠，需要在collate_fn中单独处理
        data = dict(img = img,
                    img_meta = img_meta,
                    gt_bboxes = gt_bboxes,
                    gt_labels = gt_labels)
        # 如果gt bbox数据缺失，则重新迭代随机获取一个idx的图片
        while True:
            if len(gt_bboxes) == 0:
                idx = np.random.choice(len(self.img_anns))
                self.__getitem__(idx)
            return data

    def __len__(self):
        return len(self.img_anns)


if __name__ == "__main__":
    pass 
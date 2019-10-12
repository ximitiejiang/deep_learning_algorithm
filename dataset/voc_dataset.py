#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 18:40:47 2019

@author: ubuntu
"""
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
from utils.transform import imrescale
from dataset.base_dataset import BasePytorchDataset

class VOCDataset(BasePytorchDataset):
    """VOC数据集：用于物体分类和物体检测
    2007版+2012版数据总数16551(5011 + 11540)
    主要涉及6个文件夹：
        - ImageSets:  txt主体索引文件
                * main: 所有图片名索引，以及每一类的图片名索引(可用来做其中某几类的训练)
                * Segmentation: 所有分割图片名索引
        - Annotations: xml标注文件(label,bbox)
        - JPEGImages: jpg图片文件(img)
        - labels: txt标签文件(没用到)
        - SegmentationClass: png语义分割文件(seg)
        - SegmentationObject: png实例分割文件(seg)
    输入：
        root_path: 根目录
        ann_file: 标注文件xml目录
        subset_path: 子数据集目录，比如2007， 2012
        seg_prefix: 分割数据目录，从而识别是用语义分割数据还是用实例分割数据
        difficult: 困难bbox(voc中有部分bbox有标记为困难，比如比较密集的bbox，当前手段较难分类和回归)，一般忽略
    """
    
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    
    def __init__(self,
                 root_path=None,
                 ann_file=None,
                 img_prefix=None,
                 seg_prefix=None,
                 with_difficult=False,
                 img_transform=None,
                 label_transform=None,
                 bbox_transform=None,
                 aug_transform=None,
                 seg_transform=None,
                 mode='train'):
        
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix if seg_prefix is not None else ''
        self.with_difficult = with_difficult
        # 变换函数
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.bbox_transform = bbox_transform
        self.aug_transform = aug_transform
        self.seg_transform = seg_transform
        
        # 注意：这里对标签值的定义跟常规分类不同，常规分类问题的数据集标签是从0开始，
        # 这里作为检测问题数据集标签是从1开始，目的是把0预留出来作为背景anchor的标签。
        self.class_label_dict = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}  # 从1开始(1-20). 
        # 加载图片标注表(只是标注所需文件名，而不是具体标注值)，额外加了一个w,h用于后续参考
        self.img_anns = self.load_annotations(self.ann_file) 
        # TODO: 增加在mode=train下的数据过滤
        
    
    def load_annotations(self, ann_file):
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
                img_file = self.img_prefix[i] + 'JPEGImages/{}.jpg'.format(img_id)
                xml_file = self.img_prefix[i] + 'Annotations/{}.xml'.format(img_id)
                seg_file = self.img_prefix[i] + self.seg_prefix + '{}.png'.format(img_id)
                # 解析xml文件
                # TODO: 这部分解析是否可去除，原本只是为了获得img的w,h,在img_transform里边已经搜集
                tree = ET.parse(xml_file)
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text) 
                
                img_anns.append(dict(img_id=img_id, img_file=img_file, xml_file=xml_file, width=width, height=height))
                # 补充segment file路径信息
                segmented = int(root.find('segmented').text)
                if segmented:   # 只有在包含segmented这个flag时，才存入seg file的路径，否则存入None
                    img_anns[-1].update(seg_file=seg_file)
                else:
                    img_anns[-1].update(seg_file=None)
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
        # 组合数据包为dict: 并确保bbox是float, label是int64(long)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_difficult=bboxes_difficult.astype(np.float32),
            labels_difficult=labels_difficult.astype(np.int64))
        return ann
        
    def __getitem__(self, idx):
        """虽然ann中有解析出difficult的情况，但这里简化统一没有处理difficult的情况，只对标准数据进行输出。
        """
        data = {}
        ori_shape = None
        scale_shape = None
        pad_shape = None
        scale_factor = None
        flip = None
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
        if self.bbox_transform is not None:
            gt_bboxes = self.bbox_transform(gt_bboxes, scale_shape, scale_factor, flip)
        # label transform
        if self.label_transform is not None:
            gt_labels = self.label_transform(gt_labels)
        
        # 组合img_meta
        img_meta = dict(ori_shape = ori_shape,
                        scale_shape = scale_shape,
                        pad_shape = pad_shape,
                        scale_factor = scale_factor,
                        flip = flip)
        # 组合数据: 注意img_meta数据无法堆叠，尺寸不一的img也不能堆叠，所以需要在collate_fn中自定义处理方式
        data.update(img = img,
                    img_meta = img_meta,
                    gt_bboxes = gt_bboxes,
                    gt_labels = gt_labels,
                    stack_list = ['img'])
        
        # 如果是分割任务，提供的是分割png，所以用的是seg_transform
        if self.seg_transform is not None and img_info.get('seg_file') is not None:
            seg_path = self.img_anns[idx]['seg_file']
#            seg = cv2.imread(seg_path)   # (h,w,3)
            seg = Image.open(seg_path)   # 采用PIL.Image读入图片可以直接得到用0-20类别值作为像素值的数据(还包括255白色边框)
            # 确保seg作为标签必须为int64(long)
            seg = np.asarray(seg) # (h,w)
            seg = self.seg_transform(seg, flip).long()  # 类似于对img的变换，只需输入seg，额外一个从img transform来的参数，保证与img一致
            data.update(seg = seg)
            data['stack_list'].append('seg')
        # 如果gt bbox数据缺失，则重新迭代随机获取一个idx的图片
        while True:
            if self.bbox_transform is not None and len(gt_bboxes) == 0:  # 确保seg时可以跳过
                idx = np.random.choice(len(self.img_anns))
                self.__getitem__(idx)
            return data

    def __len__(self):
        return len(self.img_anns)


if __name__ == "__main__":
    pass 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 07:36:02 2019

@author: ubuntu
"""
import xml.etree.ElementTree as ET
from dataset.coco_dataset import CocoDataset
from dataset.voc_dataset import VOCDataset


class CityScapesDataset(VOCDataset):
    """主要是分割数据集，特点是以道路交通为主，且包含一部分非常精细标注的数据
    所下载的数据集不是官方数据集，而是经过处理的，来自https://blog.csdn.net/zym19941119/article/details/81198315
    1. 可以用核心的精细标注数据做分割任务(gtFine)：这是当前数据集处理程序选择的方式，包含精细图片训练集2975张，验证集500张
        - img_ann采用trainInstances.txt
        - seg_ann采用trainLabels.txt
        
    2. 也可以考虑用普通标注的数据做分割任务(gtFine + leftImg8bit): 
        - img_ann采用trainImages.txt + trainInstances.txt
    """
    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def load_annotations(self, ann_file):
        """可加载多个标注文件
        """
        img_anns = []
        for i, af in enumerate(ann_file): # 分别读取多个子数据源
            with open(af) as f:
                img_ids = f.readlines()
            for j in range(len(img_ids)):
                img_ids[j] = img_ids[j][:-1]  # 去除最后的\n字符
            # 基于图片id打开annotation文件，获取img/xml文件名
            for img_id in img_ids:
                img_file = self.img_prefix[i] + img_id

                
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
        
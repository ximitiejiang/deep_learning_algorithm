#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 18:40:47 2019

@author: ubuntu
"""
import numpy as np
import cv2
from pycocotools.coco import COCO
from dataset.base_dataset import BasePytorchDataset

class CocoDataset(BasePytorchDataset):
    """COCO数据集：用于物体分类和物体检测
    2007版+2012版数据总数16551(5011 + 11540), 可以用2007版做小数据集试验。
    主要涉及3个文件夹：
        - ImageSets:  txt主体索引文件
                * main: 所有图片名索引，以及每一类的图片名索引(可用来做其中某几类的训练)
                * Segmentation: 所有分割图片名索引
        - Annotations: xml标注文件(label,bbox)
        - JPEGImages: jpg图片文件(img)
        - labels: txt标签文件
        - SegmentationClass: png语义分割文件(mask)
        - SegmentationObject: png实例分割文件(seg)
    输入：
        root_path: 根目录
        ann_file: 标注文件xml目录
        subset_path: 子数据集目录，比如2007， 2012
        seg_prefix: 分割数据目录，从而识别是用语义分割数据还是用实例分割数据
        difficult: 困难bbox(voc中有部分bbox有标记为困难，比如比较密集的bbox，当前手段较难分类和回归)，一般忽略
    """
    
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
    
    def __init__(self,
                 root_path=None,
                 ann_file=None,
                 img_prefix=None,
                 img_transform=None,
                 label_transform=None,
                 bbox_transform=None,
                 aug_transform=None,
                 seg_transform=None,
                 mask_transform=None,
                 mode='train'):
        
        self.ann_file = ann_file[0] if isinstance(ann_file, list) else ann_file
        self.img_prefix = img_prefix[0] if isinstance(img_prefix, list) else img_prefix

        # 变换函数
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.bbox_transform = bbox_transform
        self.aug_transform = aug_transform
        self.seg_transform = seg_transform
        self.mask_transform = mask_transform
        
        # 加载图片标注表(只是标注所需文件名，而不是具体标注值)，额外加了一个w,h用于后续参考
        self.img_anns = self.load_annotations(self.ann_file) 
        # 注意：这里对标签值的定义跟常规分类不同，常规分类问题的数据集标签是从0开始，
        # 这里作为检测问题数据集标签是从1开始，目的是把0预留出来作为背景anchor的标签。
        self.class_label_dict = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}  # 从1开始(1-20). 
        
        # 在训练模式下过滤数据集，去掉没有ann的数据
        if mode == 'train':
            valid_inds = self._filter_imgs()  # 从118287过滤到117266
            self.img_anns = np.array(self.img_anns)[valid_inds]  # list不能直接筛
        
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(value['image_id'] for value in self.coco.anns.values())
        for i, img_ann in enumerate(self.img_anns):
            if img_ann['img_id'] not in ids_with_ann: # 如果该图不在ann id列表中则跳出
                continue
            if min(img_ann['width'], img_ann['height']) >= min_size:  # 如果图片尺寸大于最小尺寸，则记录
                valid_inds.append(i)
        return valid_inds

    
    def load_annotations(self, ann_file):
        """采用coco接口读取数据信息
        """
        self.coco = COCO(ann_file)
        # 获取图片名
        self.img_ids = self.coco.getImgIds()
        img_anns = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            img_file = self.img_prefix + info['file_name']  # TODO: 检查img_file路径的合法性
            width = info['width']
            height = info['height']
            img_id = info['id']
            img_anns.append(dict(img_id=img_id, img_file=img_file, width=width, height=height))
        
        # 获取类别号，注意：coco的类别号是从1-90中间有空洞，需要转换到1-80的形式用来训练
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)}
        
        return img_anns
    
    
    def parse_ann_info(self, idx):
        # 基于一个img_id获得多个ann id, 然后解析
        img_id = self.img_anns[idx]['img_id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)  # 返回的是一个list, 每个元素是dict, 包含('segmentation','area','iscrowd','image_id','bbox','category_id','id')

        bboxes = []
        labels = []
        segs = []
        
        masks = []
        mask_polys = []
        poly_lens = []
        for i, info in enumerate(ann_info):
            # 判断ann是否合法
            if info.get('ignore', False):
                continue
            xmin, ymin, w, h = info['bbox']  # 注意，coco提供的是xmin,ymin,w,h而不是xmin/ymin/xmax/ymax，需要手动转换
            # 判断面积信息和w,h是否合法
            if info['area'] <=0 or w <1 or h < 1:
                continue
            # 判断是否为密集文件: 类似voc的difficult，这里简化处理直接忽略
            bbox = [xmin, ymin, xmin + w - 1, ymin + h - 1]
            label = self.cat2label[info['category_id']]   # (1-80)
            if info['iscrowd']:
                continue
            else:
                bboxes.append(bbox)
                labels.append(label)
            # 存放分割数据：分割数据提供了两种格式：一种是seg格式(跟原图一样大的二值图，只包含0,1)，每个bbox对应一张图，也就是每个seg只有一个实体
            # 区别于voc： voc的seg数据像素值是1-20的类别数，分割适合用交叉熵，这coco的seg像素是0-1, 适合用binary交叉熵
            # 另一种是polys(一张图上多个polys，每个poly是一个float list), polys形式预先提取，暂时在项目中不会使用。
            
            # 如果是语义分割任务，要得到的是seg: 这里预留
            if self.seg_transform is not None:
                segs.append(self.coco.annToMask(info).astype(np.int64))
                segs = [segs[i] * labels[i] for i in range(len(labels))]  #(b,)(h,w)

            # 如果是检测任务和实例分割任务用mask： 也就是提取其中跟bbox相关的部分作为bbox的target来跟bbox计算损失，
            if self.mask_transform is not None:
                masks.append(self.coco.annToMask(info).astype(np.int64))
                mask_poly = [p for p in info['segmentation'] if len(p) >= 6] # 合法的polys需要至少3个坐标点
                poly_len = [len(p) for p in mask_poly]
                mask_polys.append(mask_poly)
                poly_lens.append(poly_len)
        
        if not bboxes:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0, ), dtype=np.int64)
        else:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        # 组合数据包为dict: 并确保bbox是float, label是int64(long), seg也是int64
        ann = dict(bboxes = bboxes.astype(np.float32),
                   labels = labels.astype(np.int64))
        
        if self.seg_transform is not None:
            segs = np.stack(segs, axis=0)  # (b,h,w)
            ann.update(segs=segs)
            
        if self.mask_transform is not None:
            ann.update(masks = masks, mask_polys=mask_polys, poly_lens=poly_lens)
        
        return ann
                          
    
    def __getitem__(self, idx):
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
        
        # 如果是检测任务中的mask
        if self.mask_transform is not None:
            mask = ann_dict['masks']
            mask = self.mask_transform(mask)
            data.update(mask = mask)
            data['stack_list'].append('mask')
        # 如果是分割任务
        if self.seg_transform is not None:
            seg = ann_dict['segs']
            # TODO: 如何在seg_transform中处理seg list(coco数据集的seg提供方式是每个instance一张seg图，组成seg list)
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
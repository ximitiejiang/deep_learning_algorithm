#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:43:28 2019

@author: ubuntu
"""
from dataset.utils import get_dataset
from dataset.trafficsign_dataset import TrafficSign
from dataset.voc_dataset import VOCDataset
from dataset.coco_dataset import CocoDataset
import matplotlib.pyplot as plt
import numpy as np
from dataset.utils import vis_bbox
from tqdm import tqdm

"""
一张图片，缩小到(1333,800),相应的gt_bbox也缩小到
所以对小物体的预测，主要取决于浅层anchor的大小，
分析过程：
1. 假设retinanet针对voc数据集，在没有别的变换情况下，输出的bbox的分布主要是中等尺寸(1024-9216)和大尺寸的bbox(>9216), 小尺寸bbox(<1024)非常少
    此时retinanet的base anchors是足够覆盖voc的中等尺寸和大尺寸anchor
2. 假设retinanet针对coco数据集
"""

"""这是retinanet最浅层0层的base_anchors, 最小面积945，最大面积2485"""
ba0 = np.array([[-19.,  -7.,  26.,  14.],
                [-25., -10.,  32.,  17.],
                [-32., -14.,  39.,  21.],
                [-12., -12.,  19.,  19.],
                [-16., -16.,  23.,  23.],
                [-21., -21.,  28.,  28.],
                [ -7., -19.,  14.,  26.],
                [-10., -25.,  17.,  32.],
                [-14., -32.,  21.,  39.]])
"""retinanet第1层base_anchors, 最小面积3969，最大面积10,201"""
ba1 = np.array([[-37., -15.,  52.,  30.],
                [-49., -21.,  64.,  36.],
                [-64., -28.,  79.,  43.],
                [-24., -24.,  39.,  39.],
                [-32., -32.,  47.,  47.],
                [-43., -43.,  58.,  58.],
                [-15., -37.,  30.,  52.],
                [-21., -49.,  36.,  64.],
                [-28., -64.,  43.,  79.]])
"""retinanet第2层base_anchors, 最小面积16,109，最大面积41,209"""
ba2 = np.array([[ -75.,  -29.,  106.,   60.],
                [ -98.,  -41.,  129.,   72.],
                [-128.,  -56.,  159.,   87.],
                [ -48.,  -48.,   79.,   79.],
                [ -65.,  -65.,   96.,   96.],
                [ -86.,  -86.,  117.,  117.],
                [ -29.,  -75.,   60.,  106.],
                [ -41.,  -98.,   72.,  129.],
                [ -56., -128.,   87.,  159.]])
"""retinanet第3层base_anchors, 最小面积65,025，最大面积164,451"""
ba3 = np.array([[-149.,  -59.,  212.,  122.],
                [-196.,  -82.,  259.,  145.],
                [-255., -112.,  318.,  175.],
                [ -96.,  -96.,  159.,  159.],
                [-129., -129.,  192.,  192.],
                [-171., -171.,  234.,  234.],
                [ -59., -149.,  122.,  212.],
                [ -82., -196.,  145.,  259.],
                [-112., -255.,  175.,  318.]])
"""retinanet第4层base_anchors, 最小面积261,003，最大面积658,377"""
ba4 = np.array([[-298., -117.,  425.,  244.],
                [-392., -164.,  519.,  291.],
                [-511., -223.,  638.,  350.],
                [-192., -192.,  319.,  319.],
                [-259., -259.,  386.,  386.],
                [-342., -342.,  469.,  469.],
                [-117., -298.,  244.,  425.],
                [-164., -392.,  291.,  519.],
                [-223., -511.,  350.,  638.]])


def show_bbox(bboxes):
    """输入array"""
#    x = np.arange(-5.0, 5.0, 0.02)
#    y1 = np.sin(x)
#
#    plt.figure(1)
#    plt.subplot(211)
#    plt.plot(x, y1)
    wh = [((bb[3]-bb[1]), (bb[2]-bb[0])) for bb in bboxes]
    areas = [(bb[3]-bb[1])*(bb[2]-bb[0]) for bb in bboxes]
    print('(w,h) = ', wh)
    print('areas = ', areas, 'min area = ', min(areas), 'max area = ', max(areas))
    
    plt.plot([0,8,8,0,0],[0,0,8,8,0])
    for bbox in bboxes:
        plt.plot([bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]],
                 [bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]])

class AnalyzeDataset():
    """用于对数据集进行预分析: 
    1.如果只是检查img/bbox/label，则checkonly=True；如果要分析数据集结构，则False    
    2.关于bboxes的尺寸问题：
        当前dataset输出的img/bbox都是经过transform之后的，也就是缩减或放大过的
        (一般是放大，比如retinanet是放大到1333,800)，下面的统计输入的img/bbox/label都是一个一个data经过变换后读取过来的
        也就是说，根据transform的不同，对应的统计结果也是不同的，所有统计结果都是反映了输入到model的img/bbox的情况
    """
    
    def __init__(self, dset_name, dset_obj, checkonly=True):
        self.name = dset_name
        self.dataset = dset_obj
        self.checkonly = checkonly
        
        if self.name == 'voc':
            self.img_norm_cfg = self.dataset.datasets[0].img_norm_cfg   # dict
            self.img_infos = self.dataset.datasets[0].img_infos + \
                                self.dataset.datasets[1].img_infos
        else:
            self.img_norm_cfg = self.dataset.img_norm_cfg
            self.img_infos = self.dataset.img_infos
            
            
        if not checkonly:
            if self.name == 'voc' or self.name == 'coco':
                self.gt_labels = []
                self.gt_bboxes = []
                self.img_names = []
                for id, info in tqdm(enumerate(self.img_infos)):  # 这是voc12(11540)的训练集, 也可以用voc07 (5011)的来查看
                    data = self.dataset[id]
                    self.gt_labels.append(data['gt_labels'].data.numpy())
                    self.gt_bboxes.append(data['gt_bboxes'].data.numpy())
                    self.img_names.append(info['filename'])
            
            elif self.name == 'traffic_sign':
                # traffic sign没有直接使用ann file里边的数据，是因为要统计的是经过基本变换送入model的img/bbox的参数，而不是原始数据集参数
                # img/bbox在每种训练设置下的统计结果都不一样
#                self.gt_labels = self.dataset.gt_labels
#                self.gt_bboxes = self.dataset.gt_bboxes
#                self.img_names = self.dataset.img_names
                self.gt_labels = []
                self.gt_bboxes = []
                self.img_names = []
                for id, info in tqdm(enumerate(self.img_infos)):
                    data = self.dataset[id]
                    self.gt_labels.append(data['gt_labels'].data.numpy())
                    self.gt_bboxes.append(data['gt_bboxes'].data.numpy())
                    self.img_names.append(info['filename'])
                        
    def imgcheck(self, idx):
        """用于初步检查数据集的好坏，以及图片和bbox/label的显示"""
        # 注意：这一步会预处理img/bboxes: 有几条是影响显示的需要做逆运算(norm/chw/rgb)
        # 同时由于数据集在处理gt label时，是把class对应到(1-n)的数组，从1开始也就意味着右移了，所以显示class时要左移
        # 其他几条不影响显示不做逆变换(img缩放/bbox缩放)
        data = self.dataset[idx]    
        class_names = self.dataset.CLASSES
        mean = self.img_norm_cfg['mean']
        std = self.img_norm_cfg['std']
        filename = self.img_infos[idx]['filename']
        # 输出图像名称和图像基本信息
        print("img name: %s"% filename)
        print("img meta: %s"% data['img_meta'].data)
        # 图像逆变换
        img = data['img'].data.numpy()   # tensor to numpy
        img = img.transpose(1,2,0)       # chw to hwc
        img = (img * std) + mean         # denormalize to (0-255)
        img = img[...,[2,1,0]]           # rgb to bgr
        # 显示
        vis_bbox(img, data['gt_bboxes'].data.numpy(), 
                 labels=data['gt_labels'].data.numpy()-1,  # 显示时标签对照回来需要左移一位
                 class_names=class_names, 
                 instance_colors=None, alpha=1., 
                 linewidth=1.5, ax=None, saveto=None)
    
    
    def cluster_bbox(self, show=False):
        """对所有bbox位置进行分析，希望能够缩小图片尺寸
           结论：发现bbox遍布整个图片，无法缩小图片尺寸
        """
        if self.checkonly:
            print('Can not analyse dataset on checkonly mode, you need change checkonly mode to False.')
            return
        else:
            coord_ctr = []
            if self.name == 'traffic_sign':
                for bboxes in self.gt_bboxes:
                    x_ctr = (bboxes[0][4] + bboxes[0][0]) / 2
                    y_ctr = (bboxes[0][5] + bboxes[0][1]) / 2
        
                    coord_ctr.append([x_ctr, y_ctr])
            coord_ctr = np.array(coord_ctr)
            if show:
                plt.figure()
                plt.subplot(1,1,1)
                plt.title("cluster for all the bboxes central point")
                plt.scatter(coord_ctr[:,0], coord_ctr[:,1])
            return coord_ctr
    
    def bbox_size(self, show=False):
        """对所有bbox位置进行分析，希望能够缩小图片尺寸
        """
        # TODO: 面积分布需要进一步优化，最好按照面积大小统计分布图，单纯统计3个等级太粗了
        if self.checkonly:
            print('Can not analyse dataset on checkonly mode, you need change checkonly mode to False.')
            return
        else:
            sizes = []
            areas = []
            abnormal = []
            for id, bboxes in enumerate(self.gt_bboxes):
                for bbox in bboxes:
    
                    if self.name == 'traffic_sign':
                        w = bboxes[0][2] - bboxes[0][0]
                        h = bboxes[0][5] - bboxes[0][1]
                    else:
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
    
                    if w == 0 or h == 0:
                        abnormal.append((id, self.img_names[id]))
                        continue
                    elif (h / w > 6) or (w / h > 6):
                        abnormal.append((id, self.img_names[id]))
                        continue
                    area = w * h
                    sizes.append([w, h])
                    areas.append(area)
            sizes = np.array(sizes)
            
            areas = np.array(areas)
            bin_values = [32*32, 96*96]   # based on coco criteria
            none = len(areas[areas==0])
            small = len(areas[areas<=bin_values[0]]) - none
            big = len(areas[areas>bin_values[1]])
            medium = len(areas[(areas>bin_values[0]) & (areas<=bin_values[1])])
            
            if show:
                plt.figure()
                plt.subplot(131)
                plt.title("bboxes width and height")
                plt.xlim((0,max(sizes[:,0])+1))  # w
                plt.ylim((0,max(sizes[:,1])+1))  # h
                plt.scatter(sizes[:,0], sizes[:,1])
                
                plt.subplot(132)
                plt.title("bbox area summary(0-32*32, 32*32-96*96, 96*96-)")
                plt.bar([1,2,3], [small, medium, big])
                
                plt.subplot(133)
                plt.title("bbox area summary")
                plt.scatter(np.arange(len(areas)), areas)
                
    #            for id,name in abnormal:
    #                data = self.dataset[id]
    #                vis_bbox(data['img'], data['gt_bboxes'], labels=data['gt_labels'], class_names=self.dataset.CLASSES, 
    #                         instance_colors=None, alpha=1., linewidth=1.5, ax=None, saveto=None)
            return sizes, abnormal
        
    
    def types_bin(self, show=False):
        """对所有bbox的类型进行分析，看是否有类别不平衡问题
           类别从0到20总共21类
        """
        if self.checkonly:
            print('Can not analyse dataset on checkonly mode, you need change checkonly mode to False.')
            return
        else:
            gts_dict = {0:0} 
            for gts in self.gt_labels:
                for gt in gts:
                    if gt in gts_dict.keys():
                        key = gt
                        gts_dict[key] += 1
                    else:
                        key = gt
                        gts_dict[key] = 1
            if show:
                print("\n")
    #            print(gts_dict)
                bar_x = gts_dict.keys()
                bar_y = gts_dict.values()
    #            bar_values = [gts_dict[gt] for gt in range(21)]
                plt.figure()
                plt.title("nums for each classes")
                plt.bar(bar_x, bar_y)
            return gts_dict
    
    def bbox_kmean(self, k=6):
        pass
    
if __name__ == '__main__':
    
    dset = 'trafficsign'
    
    if dset == 'trafficsign':
        dataset_type = 'TrafficSign'    # 改成trafficsign
        data_root = './data/traffic_sign/'  # 改成trafficsign
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], 
                            std=[58.395, 57.12, 57.375], 
                            to_rgb=True)
        trainset_cfg=dict(
                    type=dataset_type,
                    ann_file=data_root + 'train_label_fix.csv',
                    img_prefix=data_root + 'Train_fix/',
                    img_scale=(1333, 800),
                    img_norm_cfg=img_norm_cfg,
                    size_divisor=32,
                    with_label=True,
                    extra_aug=None)
        dataset = get_dataset(trainset_cfg, TrafficSign)
        ana = AnalyzeDataset('traffic_sign', dataset, checkonly=False)
        ana.imgcheck(2)
#        ana.cluster_bbox(show=True)       
        ana.types_bin(show=True)
        ana.bbox_size(show=True)
    
    if dset == 'voc':
        dataset_type = 'VOCDataset'
        data_root = './data/VOCdevkit/'
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], 
                std=[58.395, 57.12, 57.375], 
                to_rgb=True)
        trainset_cfg=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            img_scale=(1333, 800),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_crowd=False,
            with_label=True)
        
        dataset = get_dataset(trainset_cfg, VOCDataset)
        ana = AnalyzeDataset('voc', dataset, checkonly=True)  # 如果只是显示图片，则checkonly=True；如果要分析数据集结构，则False
        ana.imgcheck(120)
        
        ana.types_bin(show=True)
        ana.bbox_size(show=True)

    if dset == 'coco':
        dataset_type = 'CocoDataset'
        data_root = './data/coco/'
        trainset_cfg=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            img_scale=(300, 300),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            resize_keep_ratio=False)
        
        dataset = get_dataset(trainset_cfg, CocoDataset)
        ana = AnalyzeDataset('coco', dataset, checkonly=True)
        ana.imgcheck(21)
        
        ana.types_bin(show=True)
        ana.bbox_size(show=True)
            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:43:28 2019

@author: ubuntu
"""
from dataset.utils import get_dataset, ConcatDataset
from dataset.trafficsign_dataset import TrafficSign
from dataset.voc_dataset import VOCDataset
from dataset.coco_dataset import CocoDataset
import matplotlib.pyplot as plt
import numpy as np
from dataset.utils import vis_bbox
from tqdm import tqdm
from sklearn.cluster import KMeans

"""
一张图片，缩小到(1333,800),相应的gt_bbox也缩小到
所以对小物体的预测，主要取决于浅层anchor的大小，
分析过程：
1. anchor的尺寸由3个参数决定：
    + base_size=(8,16,32,64,128)这个参数作为一个基础参数，是base anchor的参考尺寸，取值采用感受野大小，也是anchor的最小尺寸目的就是至少覆盖感受野，再小就连感受野都覆盖不住了。
    + anchor_ratio=(0.5,1,2)这个参数一般不变，目的是能够捕捉了h/w比例不一的物体
    + anchor_scale=()这个参数用于在base_size基础上放大，这也是最影响anchor尺寸的因子

2. 小物体，中物体，大物体的区分：基于coco数据集的划分方式，面积<32*32为小物体，面积>96*96为打物体，中间的就是中物体
    + 简单点就是面积在1000-9000之间的中物体，两头就是小物体和大物体

2. 分析voc数据集bbox的分布情况：(基于scale到(1333,800)这个最常见的尺寸作为分析基础且不做数据增强，retinanet/fasterrcnn/cascadercnn都是用这个尺寸的图片)
   可查看voc数据集在retinanet的transform基础上bbox的分布情况(voc_dataset_summary.png)，发现bbox主要分布在大于9216(96*96)的区间，属于大物体类型数据集
   大物体区间(>96*96)：bbox最大到800,000(900*900)，而anchor最大到658,377，差不多可以满足大部分要求
   中物体区间(>32*32)：bbox个数较少，且被0层/1层的anchor完美覆盖，
   小物体区间(<32*32)：transform后没有bbox落在小物体区间，所以虽然anchor无法覆盖，但对精度影响不大
"""

"""retinanet anchor: anchor_base = [8,16,32,64,128], anchor_scales = [4., 5., 6.35], anchor_ratios = [0.5, 1., 2.]
    原有anchor范围：(32*32, 40*40, 50*50), (64*64, 80*80, 101*101), (128*128, 160*160, 203*203), (256*256, 320*320, 406*406), (512*512, 640*640, 812*812) 
"""
re_ba0 = np.array([[-19.,  -7.,  26.,  14.],  # 面积945 - 2485
                   [-25., -10.,  32.,  17.],
                   [-32., -14.,  39.,  21.],
                   [-12., -12.,  19.,  19.],
                   [-16., -16.,  23.,  23.],
                   [-21., -21.,  28.,  28.],
                   [ -7., -19.,  14.,  26.],
                   [-10., -25.,  17.,  32.],
                   [-14., -32.,  21.,  39.]])
re_ba1 = np.array([[-37., -15.,  52.,  30.],  # 面积3969 - 10,201
                   [-49., -21.,  64.,  36.],
                   [-64., -28.,  79.,  43.],
                   [-24., -24.,  39.,  39.],
                   [-32., -32.,  47.,  47.],
                   [-43., -43.,  58.,  58.],
                   [-15., -37.,  30.,  52.],
                   [-21., -49.,  36.,  64.],
                   [-28., -64.,  43.,  79.]])
re_ba2 = np.array([[ -75.,  -29.,  106.,   60.],   # 面积16,109 - 41,209
                   [ -98.,  -41.,  129.,   72.],
                   [-128.,  -56.,  159.,   87.],
                   [ -48.,  -48.,   79.,   79.],
                   [ -65.,  -65.,   96.,   96.],
                   [ -86.,  -86.,  117.,  117.],
                   [ -29.,  -75.,   60.,  106.],
                   [ -41.,  -98.,   72.,  129.],
                   [ -56., -128.,   87.,  159.]])
re_ba3 = np.array([[-149.,  -59.,  212.,  122.],  # 面积65,025 - 164,451
                   [-196.,  -82.,  259.,  145.],
                   [-255., -112.,  318.,  175.],
                   [ -96.,  -96.,  159.,  159.],
                   [-129., -129.,  192.,  192.],
                   [-171., -171.,  234.,  234.],
                   [ -59., -149.,  122.,  212.],
                   [ -82., -196.,  145.,  259.],
                   [-112., -255.,  175.,  318.]])
re_ba4 = np.array([[-298., -117.,  425.,  244.],  # 面积261,003 - 658,377
                   [-392., -164.,  519.,  291.],
                   [-511., -223.,  638.,  350.],
                   [-192., -192.,  319.,  319.],
                   [-259., -259.,  386.,  386.],
                   [-342., -342.,  469.,  469.],
                   [-117., -298.,  244.,  425.],
                   [-164., -392.,  291.,  519.],
                   [-223., -511.,  350.,  638.]])
retinanet_ba_list = [re_ba0, re_ba1, re_ba2, re_ba3, re_ba4]  # [arry1, array2, ...]
# ----------------------------------------------------------------------------
"""cascade rcnn anchor: anchor_base = [4,8,16,32,64], anchor_scales = [8], anchor_ratios = [0.5, 1., 2.]
    原有anchor分割范围是32*32, 64*64, 128*128, 256*256, 512*512
    scale调为5分割范围是20*20, 40*40, 80*80, 160*160, 320*320
    scale调为4分割范围是16×16, 32×32, 64×64, 128×128, 256×256 (采用这种方式，前两档正好跟k=2的kmean算法算出来的2个点差不多(16.4,16.3) (25.9,34.7)
"""
cr_ba0 = np.array([[-21.,  -9.,  24.,  12.],       # 面积945 - 961
                   [-14., -14.,  17.,  17.],
                   [ -9., -21.,  12.,  24.]])    
cr_ba1 = np.array([[-41., -19.,  48.,  26.],       # 面积3969 - 4005
                   [-28., -28.,  35.,  35.],
                   [-19., -41.,  26.,  48.]])
cr_ba2 = np.array([[-83., -37.,  98.,  52.],       # 面积16109 - 16129
                   [-56., -56.,  71.,  71.],
                   [-37., -83.,  52.,  98.]])
cr_ba3 = np.array([[-165.,  -75.,  196.,  106.],   # 面积65025 - 65341
                   [-112., -112.,  143.,  143.],
                   [ -75., -165.,  106.,  196.]])
cr_ba4 = np.array([[-330., -149.,  393.,  212.],   # 面积261003 - 261121
                   [-224., -224.,  287.,  287.],
                   [-149., -330.,  212.,  393.]])
cascadercnn_ba_list = [cr_ba0, cr_ba1, cr_ba2, cr_ba3, cr_ba4]
#-----------------------------------------------------------------------------

def show_bbox(bboxes):
    """输入array"""
    assert isinstance(bboxes, np.ndarray), 'bboxes should be ndarray.'  
    wh = [((bb[3]-bb[1]), (bb[2]-bb[0])) for bb in bboxes]
    areas = [(bb[3]-bb[1])*(bb[2]-bb[0]) for bb in bboxes]
    print('(w,h) = ', wh)
    print('areas = ', areas, 'min area = ', min(areas), 'max area = ', max(areas))
    
    for bbox in bboxes:
        plt.plot([bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]],
                 [bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]])
# debug
#show_bbox(cascadercnn_ba_list[4])

class AnalyzeBbox():
    def __init__(self, bbox_list):
#        self.ana = AnalyzeDataset(dset_name, dset_obj, checkonly=True)
        assert isinstance(bbox_list, list), 'bbox_list should be a list of array'
        self.bbox_list = bbox_list
        
    
    def bbox_summary(self):
        for level, bboxes in enumerate(self.bbox_list):
            wh = [((bb[3]-bb[1]), (bb[2]-bb[0])) for bb in bboxes]
            areas = [(bb[3]-bb[1])*(bb[2]-bb[0]) for bb in bboxes]
            print('level %d areas: '%level, areas)
            print('min area = ', min(areas), 'max area = ', max(areas))
    
    def bboxshow_all(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for level, bboxes in enumerate(self.bbox_list):
            self.bboxshow_single(bboxes, ax=ax)
        
    """ax.add_patch()一般用来绘制Rectangle/Cirle/Polygon：先创建ax(ax=fig.add_subplot(1,1,1)), 再创建rect(plt.Rectangle(), 最后添加进ax(ax.add_patch())"""
    def bboxshow_single(self, bboxes, ax=None, img=None):
        """显示一组bbox array"""
        assert isinstance(bboxes, np.ndarray), 'bboxes should be ndarray.'    
#        plt.plot([0,8,8,0,0],[0,0,8,8,0])  # 绘制base size box
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        for bbox in bboxes:
            ax.plot([bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]],
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
              
        if isinstance(self.dataset, ConcatDataset):  # voc是concatedataset需要单独处理
            self.img_norm_cfg = self.dataset.datasets[0].img_norm_cfg   # dict
            self.img_infos = self.dataset.datasets[0].img_infos + \
                                self.dataset.datasets[1].img_infos
            self.img_scale = self.dataset.datasets[0].img_scales
        else:
            self.img_norm_cfg = self.dataset.img_norm_cfg
            self.img_infos = self.dataset.img_infos
            self.img_scale = self.dataset.img_scales
                       
        if not checkonly:
            self.ana_range = 1000    # debug: 可定义分析的数据集范围(避免分析整个数据集导致太慢)
        
            self.gt_labels = []
            self.gt_bboxes = []
            self.img_names = []
            for id, info in tqdm(enumerate(self.img_infos[:self.ana_range])):  # 这是voc12(11540)的训练集, 也可以用voc07 (5011)的来查看
                data = self.dataset[id]
                self.gt_labels.append(data['gt_labels'].data.numpy())
                self.gt_bboxes.append(data['gt_bboxes'].data.numpy())
                self.img_names.append(info['filename'])
            
#            elif self.name == 'traffic_sign':
#                # traffic sign没有直接使用ann file里边的数据，是因为要统计的是经过基本变换送入model的img/bbox的参数，而不是原始数据集参数
#                # img/bbox在每种训练设置下的统计结果都不一样
##                self.gt_labels = self.dataset.gt_labels
##                self.gt_bboxes = self.dataset.gt_bboxes
##                self.img_names = self.dataset.img_names
#                self.gt_labels = []
#                self.gt_bboxes = []
#                self.img_names = []
#                for id, info in tqdm(enumerate(self.img_infos[:self.ana_range])):   # 这里可以控制数据集分析的个数，开始先分析1000个以内，避免跑很长时间
#                    data = self.dataset[id]
#                    self.gt_labels.append(data['gt_labels'].data.numpy())
#                    self.gt_bboxes.append(data['gt_bboxes'].data.numpy())
#                    self.img_names.append(info['filename'])
                        
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
            raise ValueError('Can not analyse dataset on checkonly mode, you need change checkonly mode to False.')

        else:
            coord_ctr = []
            if self.name == 'traffic_sign':
                for bboxes in self.gt_bboxes:
                    x_ctr = (bboxes[0][4] + bboxes[0][0]) / 2
                    y_ctr = (bboxes[0][5] + bboxes[0][1]) / 2
        
                    coord_ctr.append([x_ctr, y_ctr])
            coord_ctr = np.array(coord_ctr)
            if show:
                title = 'Dataset: ' + self.name + '(after ImgTransform)' \
                        + ' with range ' + str(self.ana_range) \
                        + ' and size ' +  str(self.img_scale)
                plt.figure()
                plt.suptitle(title)
                
                plt.subplot(1,1,1)
                plt.title("cluster for all the bboxes central point")
                plt.scatter(coord_ctr[:,0], coord_ctr[:,1])
            return coord_ctr
    
    def bbox_size(self, show=False):
        """对所有bbox的w/h，面积范围，面积分布 一起进行分析
        """
        if self.checkonly:
            raise ValueError('Can not analyse dataset on checkonly mode, you need change checkonly mode to False.')

        else:
            sizes = []
            areas = []
            abnormal = []
            for id, bboxes in enumerate(self.gt_bboxes):
                for bbox in bboxes:
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
            sizes = np.array(sizes)   # (m, 2)
            centers = self.data_kmean(sizes, k=5)  # k=5, 则在5层特征层都搜索
            
            areas = np.array(areas)
            areas_100_100 = areas[areas<100*100]
            bin_values = [32*32, 96*96]   # based on coco criteria
            none = len(areas[areas==0])
            small = len(areas[areas<=bin_values[0]]) - none
            big = len(areas[areas>bin_values[1]])
            medium = len(areas[(areas>bin_values[0]) & (areas<=bin_values[1])])
            
            if show:
                title = 'Dataset: ' + self.name + '(after ImgTransform)' \
                        + ' with range ' + str(self.ana_range) \
                        + ' and size ' +  str(self.img_scale)
                plt.figure()
                plt.suptitle(title)
                plt.subplot(221)
                plt.title("bboxes width and height")
                plt.xlim((0,max(sizes[:,0])+1))  # w
                plt.ylim((0,max(sizes[:,1])+1))  # h
                plt.scatter(sizes[:,0], sizes[:,1])
                plt.scatter(centers[:,0], centers[:,1], s=50, c='r')
                print('kmean of %d points are: '%len(centers), centers)
                
                plt.subplot(222)
                plt.title("bbox area summary(0-32*32, 32*32-96*96, 96*96-)")
                plt.bar([1,2,3], [small, medium, big])
                
                plt.subplot(223)
                plt.title("bbox area bins for all")
                nums, bins, _ = plt.hist(x=areas, bins=20)
                
                plt.subplot(224)
                plt.title("bbox area bins for small&medium(area<100*100)")
                nums, bins, _ = plt.hist(x=areas_100_100, bins=50)
                
                
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
            raise ValueError('Can not analyse dataset on checkonly mode, you need change checkonly mode to False.')
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
                title = 'Dataset: ' + self.name + '(after ImgTransform)' \
                        + ' with range ' + str(self.ana_range) \
                        + ' and size ' +  str(self.img_scale)
                plt.figure()
                plt.suptitle(title)
                plt.title("nums for each classes")
                plt.bar(bar_x, bar_y)
            return gts_dict
    
    def data_kmean(self, data, k=3):
        """通过kmean算法找到数据集bbox的kmean，可以检查bbox的w,h两个特征向量的kmean
        其中k=3(比如cascade rcnn)或k=9(比如retinanet)
        得到的k可理解为对应的每组anchor的方形框面积，从而开发就可得到
        比如针对cascade rcnn原来的anchor参数是base_size=[4,8,16,32,64], scale=[8], ratio=[0.5,1,2]
        如果kmean得到area=400, 则通过base_size=[4,8,16,32,64], area=base_size*anchor_scale*ratio(1)
        所以计算得到w=h=sqrt(400)/4 = 5, 也就可以取scale=[5]
        """
        assert isinstance(data, np.ndarray), 'the input data should be ndarray.'  # (m,2) for w, h
        kmean = KMeans(n_clusters=k)
        kmean.fit(data)
        
        centers = kmean.cluster_centers_
                    
        return centers
    
if __name__ == '__main__':
    
    dset = 'trafficsign'  # voc / coco / trafficsign / bbox
    
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
#                    img_scale=(1333, 800),      # 把图片缩小
                    img_scale=(3200, 1800),      # 把图片保持尺寸
                    img_norm_cfg=img_norm_cfg,
                    size_divisor=32,
                    with_label=True,
                    extra_aug=None)
        dataset = get_dataset(trainset_cfg, TrafficSign)
        ana = AnalyzeDataset('traffic_sign', dataset, checkonly=False)
#        ana.cluster_bbox(show=True)       
        ana.types_bin(show=True)
        ana.bbox_size(show=True)
#        ana.imgcheck(2)
    
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
        ana = AnalyzeDataset('voc', dataset, checkonly=False)  # 如果只是显示图片，则checkonly=True；如果要分析数据集结构，则False       
        ana.types_bin(show=True)
        ana.bbox_size(show=True)
#        ana.imgcheck(120)

    if dset == 'coco':
        dataset_type = 'CocoDataset'
        data_root = './data/coco/'
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], 
                std=[58.395, 57.12, 57.375], 
                to_rgb=True)
        trainset_cfg=dict(
#            type=dataset_type,   # 这部分是基于retinanet的预处理
#            ann_file=data_root + 'annotations/instances_train2017.json',
#            img_prefix=data_root + 'train2017/',
#            img_scale=(1333, 800),
#            img_norm_cfg=img_norm_cfg,
#            size_divisor=32,
#            flip_ratio=0,
#            with_mask=False,
#            with_crowd=False,
#            with_label=True)
            
            type=dataset_type,    # 这部分是基于ssd的预处理
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            img_scale=(300, 300),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0,
            with_mask=False,
            with_crowd=False,
            with_label=True)
        
        dataset = get_dataset(trainset_cfg, CocoDataset)
        ana = AnalyzeDataset('coco', dataset, checkonly=False)
        ana.types_bin(show=True)
        ana.bbox_size(show=True)
#        ana.imgcheck(21)
    
    if dset == 'bbox':
        anb = AnalyzeBbox(cascadercnn_ba_list)
        anb.bbox_summary()
        anb.bboxshow_all()
        anb.bboxshow_single(retinanet_ba_list[0])
            
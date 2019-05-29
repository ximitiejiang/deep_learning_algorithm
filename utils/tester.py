#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 08:26:28 2019

@author: ubuntu
"""

import torch
import cv2
import numpy as np
import csv
import mmcv
from tqdm import tqdm

from model.checkpoint import load_checkpoint
from utils.config import Config
from dataset.transforms import ImageTransform
from dataset.class_names import get_classes
from dataset.utils import vis_bbox, opencv_vis_bbox
from model.one_stage_detector import OneStageDetector
from model.parallel.data_parallel import NNDataParallel
from dataset.utils import get_dataset
from dataset.utils import build_dataloader
from utils.map import eval_map
import sys,os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)

class Tester(object):
    """测试基类，用于进行单图/多图/摄像头测试
    1. cfg/model
    2. data
    3. run_single
    4. run
    """        
    def __init__(self, config_file, model_class, weights_path, 
                 dataset_name='voc', device = 'cuda:0'):
        self.config_file = config_file
        self.model_class = model_class
        self.weights_path = weights_path
        self.class_names = get_classes(dataset_name)
        self.device = device
        
        self.init_cfg_model()  # generate self.cfg, self.model
        
    def init_cfg_model(self):
        """准备cfg,model
        """
        # 1. 配置文件
        self.cfg = Config.fromfile(self.config_file)
        self.cfg.model.pretrained = None     # eval模式不再加载backbone的预训练参数，因为都在后边的checkpoints里边包含了。通过model load checkpoint统一来操作。
        # 2. 模型
        self.model = self.model_class(self.cfg)
        _ = load_checkpoint(self.model, self.weights_path)
        self.model = self.model.to(self.device)
        self.model.eval()             
    
    def preprocess_data(self, cfg, img, transformer=None):
        ori_shape = img.shape
        if transformer is None:
            transformer = ImageTransform(**cfg.img_norm_cfg, size_divisor=cfg.data.test.size_divisor)   # 原来只用了img_norm_cfg，自己增加了size_divisor
            
        img, img_shape, pad_shape, scale_factor = transformer(
            img, 
            scale= cfg.data.test.img_scale, 
            keep_ratio=False)  
            # 测试时没有额外别的预处理，所以只是把图像做一个基本缩放，由于要缩放到制定尺寸以内，假如保证比例则有可能图片尺寸
            # 区别：ssd要求输入必须300*300，所以keep_ratio必须False而使用resize来调节图片尺寸，否则可能导致图片在rescale的
            #      retinanet要求输入在1333*800以内，所以keep_ratio通常为True而使用rescale()来调节图片尺寸，让图片上物体能够保持原始尺寸比例
        img = torch.tensor(img).to(self.device).unsqueeze(0) 
        
        # 4. 数据包准备
        img_meta = [dict(ori_shape=ori_shape,
                         img_shape=img_shape,
                         pad_shape=pad_shape,
                         scale_factor = scale_factor,
                         flip=False)]
    
        data = dict(img=[img], img_meta=[img_meta])
        return data
    
    def run_single(self, ori_img, data, show=True, saveto=None, bboxes=None):
        """对单张图片计算结果
        Args:
            data
            bboxes(m,4): 可以单独输入bboxes用来绘制额外的bboxes
        """ 
        with torch.no_grad():
            result = self.model(**data, return_loss=False, rescale=True)  # (20,)->(n,5)or(0,5)->(xmin,ymin,xmax,ymax,score)         
        # 提取labels
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) 
                    for i, bbox in enumerate(result)]    # [(m1,), (m2,)...]
        labels = np.concatenate(labels)  # (m,)
        bboxes = np.vstack(result)       # (m,5)
        scores = bboxes[:,-1]            # (m,)
        
        single_results = [bboxes]
        single_results.append(labels)
        single_results.append(scores)
        
        if show: # 使用show变量区分是使用vis_bbox还是使用opencv_vis_bbox，用opencv_vis_bbox处理视频和cam
            vis_bbox(
                ori_img.copy(), *single_results, score_thr=0.3, class_names=self.class_names, saveto=saveto)
            # opencv版本的显示效果不太好，用matplotlib版本的显示文字较好
#            opencv_vis_bbox(
#                img.copy(), *single_results, score_thr=0.2, class_names=self.class_names, saveto=saveto)

        return single_results   # list [ array, array, array]
            
    def run(self, img_path):
        raise NotImplementedError('run() function not implemented!')
                

class TestImg(Tester):
    """用于图片的检测，可输入单张图片，也可输入多张图片list"""
    
    def __init__(self, config_file, model_class, weights_path, 
                 dataset_name='voc', device = 'cuda:0'):
        super().__init__(config_file, model_class, weights_path, 
                 dataset_name, device = 'cuda:0')
    
    def run_single(self, ori_img, data, show=True, saveto=None):
        """对单张图片计算结果
        Args:
            data
            bboxes(m,4): 可以单独输入bboxes用来绘制额外的bboxes
        """ 
        with torch.no_grad():
            result = self.model(**data, return_loss=False, rescale=True)  # (20,)->(n,5)or(0,5)->(xmin,ymin,xmax,ymax,score)         

        bboxes = np.vstack(result)       # (m,5)
        single_results = [bboxes]
        # 提取labels
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) 
                    for i, bbox in enumerate(result)]    # [(m1,), (m2,)...]
        labels = np.concatenate(labels)  # (m,)
        scores = bboxes[:,-1]            # (m,)
        
        single_results.append(labels)
        single_results.append(scores)
        
        if show: # 使用show变量区分是使用vis_bbox还是使用opencv_vis_bbox，用opencv_vis_bbox处理视频和cam
            vis_bbox(
                ori_img.copy(), *single_results, score_thr=0.3, class_names=self.class_names, saveto=saveto)
            # opencv版本的显示效果不太好，用matplotlib版本的显示文字较好
#            opencv_vis_bbox(
#                img.copy(), *single_results, score_thr=0.2, class_names=self.class_names, saveto=saveto)

        return single_results   # list [ array, array, array]
    
    def run(self, img_path):
        if isinstance(img_path, str):
            # img
            ori_img = cv2.imread(img_path)
            data = self.preprocess_data(self.cfg, ori_img)
            # run
            result_name = os.path.join(img_path[:-4] + '_result.jpg')
            _ = self.run_single(ori_img, data, show=True, saveto=result_name)
         
        elif isinstance(img_path, list):
            for i, p in enumerate(img_path):
                assert isinstance(p, str), "img_path content should be str."
                ori_img = cv2.imread(p)
                data = self.preprocess_data(self.cfg, ori_img)
                result_name = os.path.join(p[:-4] + '_result.jpg')
                _ = self.run_single(ori_img, data, show=True, saveto=result_name)
        else:
            raise TypeError("path type should be str for one img or list for multiple imgs.")
    
    @staticmethod
    def inside_imshow(img_path, bboxes=None, scale_factor=None, scores=None):
        """显示内部的bbox在原始图片上的效果，输入的bboxes为5列(xmin,ymin,xmax,ymax, score)
        """
        if isinstance(img_path, str):
            ori_img = cv2.imread(img_path)
        if bboxes is None:
            bboxes = np.zeros((4,))
        else:
            scores = bboxes[:, 4]
            bboxes = bboxes[:,:4]

        # 如果是scale状态的bbox，则需要除以scale变回原图尺寸，但如果是multiclass_nms则已经是恢复scale过了。
        if scale_factor is not None:
            assert isinstance(scale_factor, np.ndarray), 'scale_factor should be ndarray(4,)'
            bboxes /= scale_factor   # 把bbox放大回原图尺寸(这里是假定图像在test阶段只做了scale一个transform，没有其他影响尺寸的transform)
        vis_bbox(ori_img.copy(), bboxes, scores=scores, instance_colors=None, alpha=1., linewidth=0.5)
        

class TestImgResultGenerator(Tester):
    """用于测试一组图片，并把测试结果写入csv文件"""
    def __init__(self, config_file, model_class, weights_path, 
                 dataset_name='voc', device = 'cuda:0'):
        super().__init__(config_file, model_class, weights_path, 
                 dataset_name, device = 'cuda:0')
    
    def run_single(self, ori_img, data, show=True, saveto=None):
        """对单张图片计算结果
        注意：由于之前标签class都是从1开始(在数据集的class2label里)，multiclass_nms()里边生成labels时再减一个1平衡
        但这里针对traffic sign数据集，labels从0到20，所以这里手动增加一个1,才能得到正确的label预测
        """ 
        with torch.no_grad():
            result = self.model(**data, return_loss=False, rescale=True)  # (20,)->(n,5)or(0,5)->(xmin,ymin,xmax,ymax,score)         
        # 提取labels
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) 
                    for i, bbox in enumerate(result)]    # [(m1,), (m2,)...]
        labels = np.concatenate(labels)  # (m,)
        bboxes = np.vstack(result)       # (m,5)
        scores = bboxes[:,-1]            # (m,)
        
        single_results = [bboxes]
        single_results.append(labels)
        single_results.append(scores)
                     
        if len(single_results[0])==0:    # 如果结果为空，说明没有预测类，此时令标签=0(其他)
            single_results[0] = np.zeros((1,5))
            single_results[1] = np.zeros((1,))
            single_results[2] = np.zeros((1,))
        else:
            single_results[1] = single_results[1] + 1  # label预测手动+1
        if show: # 使用show变量区分是使用vis_bbox还是使用opencv_vis_bbox，用opencv_vis_bbox处理视频和cam
            vis_bbox(
                ori_img.copy(), *single_results, score_thr=0.3, class_names=self.class_names, saveto=saveto)
            # opencv版本的显示效果不太好，用matplotlib版本的显示文字较好
#            opencv_vis_bbox(
#                img.copy(), *single_results, score_thr=0.2, class_names=self.class_names, saveto=saveto)

        return single_results   # list [ array, array, array]
    
    def result2submit(self, single_results, single_path):
        """单张图片结果转换: 筛选置信度最高的结果作为提交结果
        
        single_results(list):    [bboxes/scores_array, labels_array, scores_array], 
                                with size of [(m,5), (m,), (m,)]
        single_path(str):       str of single img path
        submit(list):           [filename, x1,y1,x2,y2,x3,y3,x4,y4,score,]
        """
        submit=[]
        submit.append(single_path.split('/')[-1])        # path to name

        result = np.concatenate([single_results[0], 
                                 single_results[1].reshape(-1,1)], axis=1)   # (m, 6)
        
        max_id = np.argmax(result[:,4], axis=0)  # 0-3列为bbox,4列为scores, 5列为scores
        max_out = result[max_id]   # (1,6)
        submit.extend([int(max_out[0]),int(max_out[1]),
                       int(max_out[2]),int(max_out[1]),
                       int(max_out[2]),int(max_out[3]),
                       int(max_out[0]),int(max_out[3]), 
                       int(max_out[5])])  
            
        return submit
    
    def run(self, img_path, show=False, save=False, writeto=None):
        """ 
        Args: img_path(list of str): list of all the img path, also can be one img path in a list, like [str]
        """
        assert isinstance(img_path, list), 'the img path should be list of paths.'
        all_results = [['filename','X1','Y1','X2','Y2','X3','Y3','X4','Y4','type']]
        for i, p in tqdm(enumerate(img_path)):
            assert isinstance(p, str), "img_path content should be str."
            ori_img = cv2.imread(p)
            data = self.preprocess_data(self.cfg, ori_img)
            if save:
                result_name = os.path.join(p[:-4] + '_result.jpg')
            else:
                result_name = None
            results = self.run_single(ori_img, data, show=show, saveto=result_name)

            submit = self.result2submit(results, p)
            all_results.append(submit)
        
        if writeto is not None:
            # firstly clear all the lines in the csv file
            # then write result to csv file
            assert os.path.exists(writeto), 'the target writeto file does not exist.'
            with open(writeto, 'w', newline='') as f:
                writer = csv.writer(f)
                # 单行写入用writer.writerow([]), 多行写入用writer.writerows([[],[],...])
                writer.writerows(all_results)
            

class TestVideo(Tester):
    """用于视频或者摄像头的检测"""
    def __init__(self, config_file, model_class, weights_path, 
                 dataset_name='voc', device = 'cuda:0'):       
        super().__init__(config_file, model_class, weights_path, 
                 dataset_name, device)        
        
    def run(self, source):
        """"""
        # source can be int(as cam_id) or str(as video path)
        if isinstance(source, int):
            cam_id = source
            capture = cv2.VideoCapture(cam_id)
        elif isinstance(source, str):
            capture = cv2.VideoCapture(source)
        assert capture.isOpened(), 'Cannot capture source'
        
#        cfg, model = self.init_cfg_model()
        
        while True:
            ret, img = capture.read()
            if not ret:  # failure to read or run to end frame
                cv2.destroyAllWindows()
                capture.release()
                break
            data = self.preprocess_data(self.cfg, img)
            
            all_results = self.run_single(img, data, show=False, saveto=None) # 用run_single只获得结果不显示，而用open_vis_bbox显示
            opencv_vis_bbox(img.copy(), *all_results, score_thr=0.5, class_names=self.class_names, 
                            instance_colors=None, thickness=2, font_scale=0.5,
                            show=True, win_name='cam', wait_time=0, saveto=None)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # break by click 'q'
                cv2.destroyAllWindows()
                capture.release()
                break


class TestDataset(Tester):
    """用于在数据集上进行模型评估"""
    def __init__(self, config_file, model_class, weights_path, out_file,
                 dataset_name='voc', device = 'cuda:0'): 
        super().__init__(config_file, model_class, weights_path, 
             dataset_name, device)
        
        self.weights_path = weights_path
        self.out_file = out_file
        
    def init_cfg_model(self):
        """准备cfg,model
        """
        # 1. 配置文件
        self.cfg = Config.fromfile(self.config_file)
        self.cfg.model.pretrained = None     # eval模式不再加载backbone的预训练参数，因为都在后边的checkpoints里边包含了。通过model load checkpoint统一来操作。
        # 2. 模型
        self.cfg.gpus = 1
        self.model = self.model_class(self.cfg)
        _ = load_checkpoint(self.model, self.weights_path)
        self.model = NNDataParallel(self.model, device_ids=[0])
        self.model.eval()  
    
    def preprocess_data(self, dataset_class):
        self.dataset = get_dataset(self.cfg.data.test, dataset_class)
        self.dataloader = build_dataloader(self.dataset,
                                           imgs_per_gpu=1,
                                           workers_per_gpu=self.cfg.data.workers_per_gpu,
                                           num_gpus=1,
                                           dist=False,
                                           shuffle=False)
    
    def run_single(self, dataset_class, show=False):
        results = []
        prog_bar = mmcv.ProgressBar(len(self.dataset))
        for i, data in enumerate(self.dataloader):
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=not show, **data)
            results.append(result)
    
            if show:
                self.model.module.show_result(data, result, self.dataset.img_norm_cfg,
                                              dataset=self.dataset.CLASSES)
    
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size):
                prog_bar.update()
        return results
    
    def voc_eval(self, result_file, iou_thr=0.5):
        """voc数据集结果评估
        """
        det_results = mmcv.load(result_file)  # 加载结果文件
        gt_bboxes = []
        gt_labels = []
        gt_ignore = []
        for i in range(len(self.dataset)):   # 读取测试集的所有gt_bboxes,gt_labels
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if 'bboxes_ignore' in ann:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        if not gt_ignore:
            gt_ignore = gt_ignore
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            dataset_name = 'voc07'
        else:
            dataset_name = self.dataset.CLASSES
            
        eval_map(
            det_results,        # (4952,) (20,) (n,5)
            gt_bboxes,          # (4952,) (n,4)
            gt_labels,          # (4952,) (n,)
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=iou_thr,
            dataset=dataset_name,
            print_summary=True)
    
    def run(self, source_class):
        """用于指定数据集的预测结果生成：
        Args:
            source_class(Class): 数据集类
        """
        self.preprocess_data(source_class)  # generate self.dataset, self.dataloader
        
        if not self.out_file.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')
        # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
            
        if os.path.exists(self.out_file):
            self.voc_eval(self.out_file, iou_thr=0.5)
        else:            
            outputs = self.run_single(source_class, show=False)  # (4870,)=n_imgs (20,)=n_classes (n,5)=n_bboxes
            print('writing results to {}'.format(self.out_file))  
            mmcv.dump(outputs, self.out_file)  # 先把模型的测试结果输出到文件中: 如果文件不存在会创建 
            self.voc_eval(self.out_file, iou_thr=0.5)

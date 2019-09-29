#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:15:27 2019

@author: ubuntu
"""
import torch
import numpy as np
import cv2
from utils.prepare_training import get_config, get_dataset, get_dataloader, get_root_model
from utils.checkpoint import load_checkpoint
from utils.transform import to_device, ImgTransform
from utils.visualization import vis_loss_acc, vis_img_bbox, vis_all_opencv
from utils.tools import accuracy
from utils.dataset_classes import get_classes

# %% 分类问题
def eval_dataset_cls(cfg_path, device=None):
    """分类问题的eval dataset: 
    等效于runner中的load_from + val，但可用来脱离runner进行独立的数据集验证
    """
    # 准备验证所用的对象
    cfg = get_config(cfg_path)
    dataset = get_dataset(cfg.valset, cfg.transform_val)
    dataloader = get_dataloader(dataset, cfg.valloader)
    model = get_root_model(cfg)
    if device is None:
        device = torch.device(cfg.load_device)
    # TODO: 如下两句的顺序
    load_checkpoint(model, cfg.load_from, device)
    model = model.to(device)
    # 开始验证
    buffer = {'acc': []}
    n_correct = 0
    model.eval()
    for c_iter, data_batch in enumerate(dataloader):
        with torch.no_grad():  # 停止反向传播，只进行前向计算
            img = to_device(data_batch['img'], device)
            label = to_device(data_batch['gt_labels'], device)
            
            y_pred = model(img)
            label = torch.cat(label, dim=0)
            acc1 = accuracy(y_pred, label, topk=1)
            buffer['acc'].append(acc1)
        # 计算总体精度
        n_correct += buffer['acc'][-1] * len(data_batch['gt_labels'])
    
    vis_loss_acc(buffer, title='eval dataset')
    print('ACC on dataset: %.3f', n_correct/len(dataset))


def predict_one_img_cls(img, cfg_path):
    """针对单个样本的预测：
    直接通过model得到结果，且支持cpu/GPU预测。
    注意：需要训练完成后，或在cfg中设置load_from，也就是model先加载训练好的参数文件。
    """
    pass
    
    

# %% 检测问题
from model.runner_lib import batch_detector
from utils.tools import save2pkl, loadvar, get_time_str
from utils.map import eval_map

def eval_dataset_det(cfg_path, 
                     load_from=None, 
                     load_device=None,
                     result_file=None):
    """检测问题的eval dataset: 
    为了便于eval，添加2个形参参数，不必常去修改cfg里边的设置
    """
    # 准备验证所用的对象
    cfg = get_config(cfg_path)
    cfg.valloader.params.batch_size = 1  # 强制固定验证时batch_size=1
    # 为了便于eval，不必常去修改cfg里边的设置，直接在func里边添加2个参数即可
    if load_from is not None:
        cfg.load_from = load_from
    if load_device is not None:
        cfg.load_device = load_device
    
    dataset = get_dataset(cfg.valset, cfg.transform_val)
    dataloader = get_dataloader(dataset, cfg.valloader)
    
    model = get_root_model(cfg)
    device = torch.device(cfg.load_device)
    load_checkpoint(model, cfg.load_from, device)
    model = model.to(device)
    # 如果没有验证过
    if result_file is None: 
        # 开始验证
        model.eval()
        all_bbox_cls = []
        for c_iter, data_batch in enumerate(dataloader):
            with torch.no_grad():  # 停止反向传播，只进行前向计算
                bbox_det = batch_detector(model, data_batch, 
                                          device, return_loss=False)
                # 显示进度
                if c_iter % 100 == 0:    
                    print('%d / %d finished predict.'%(c_iter, len(dataset)))

            all_bbox_cls.append(bbox_det)  # (n_img,)(n_class,)(k,5) 
        # 保存预测结果到文件
        filename = get_time_str() + '_eval_result.pkl'
        save2pkl(all_bbox_cls, cfg.work_dir + filename)
    # 如果有现成验证文件
    else:
        all_bbox_cls = loadvar(result_file)
    # 评估
    voc_eval(all_bbox_cls, dataset, iou_thr=0.5)
    
    
class Predictor():
    """用于对图片进行预测计算，生成待显示的数据
    src: 可以输入img or img_list
    """
    def __init__(self, cfg_path, load_from=None, load_device=None):
        # 准备验证所用的对象
        self.cfg = get_config(cfg_path)
        # 为了便于eval，不必常去修改cfg里边的设置，直接在func里边添加2个参数即可
        if load_from is not None:
            self.cfg.load_from = load_from
        if load_device is not None:
            self.cfg.load_device = load_device
        self.model = get_root_model(self.cfg)
        self.device = torch.device(self.cfg.load_device)
        load_checkpoint(self.model, self.cfg.load_from, self.device)
        self.model = self.model.to(self.device)

    def __call__(self, src):
        if isinstance(src, np.ndarray):
            src = [src]
        for img in src:
            img_data = img_loader(img, self.cfg)
            with torch.no_grad():
                bbox_det = self.model(**img_data, return_loss=False)  # (n_class,)(k,5)
                # 整理成显示需要的形式
                labels = [np.full((bbox.shape[0],), i, dtype=np.int32) for i, bbox in enumerate(bbox_det)] # 用i作为label填充，由于要获取class name,这里label改成0-19
                labels = np.concatenate(labels, axis=0)  # (m, )
                bboxes = np.concatenate(bbox_det, axis=0)  # (m,5)
                scores = bboxes[:, -1]   # (m,)
                yield (img, bboxes, scores, labels)



# %% 一些support函数
    
def img_loader(img, cfg):
    """模拟dataloader的功能，加载单张图片：基本变换+升维+数据打包
    注意：如果是不固定尺寸的图片，那么collate_fn中的padding操作还需要增加。
    """
    # TODO: 对不固定尺寸的图片，collate_fn的padding需要增加
    img_transform = ImgTransform(**cfg.transform_val.img_params)
    img, ori_shape, scale_shape, pad_shape, scale_factor, flip = img_transform(img)
    img = img[None,...]     # (c,h,w)->(1,c,h,w)
    img = img.to(torch.device(cfg.load_device))
    
    img_meta = dict(ori_shape = ori_shape,
                    scale_shape = scale_shape,
                    pad_shape = pad_shape,
                    scale_factor = scale_factor,
                    flip = flip)
    
    data = dict(imgs = img,
                img_metas = [img_meta])  # 注意这里需要放入list中模拟dataloader的collate效果
    return data
    

def voc_eval(result_file, dataset, iou_thr=0.5):
    """voc数据集结果评估
    """
    if isinstance(result_file, list):
        det_results = result_file
    elif isinstance(result_file, str):
        det_results = loadvar(result_file)  # 加载结果文件
    gt_bboxes = []
    gt_labels = []
    
    for i in range(len(dataset)):   # 读取测试集的所有gt_bboxes,gt_labels
        ann = dataset.parse_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    # 调用mAP计算模块
    dataset_name = 'voc'
    eval_map(
        det_results,        # (4952,) (20,) (n,5)
        gt_bboxes,          # (4952,) (n,4)
        gt_labels,          # (4952,) (n,)
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)
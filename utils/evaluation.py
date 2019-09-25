#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:15:27 2019

@author: ubuntu
"""
import torch
import numpy as np
from utils.prepare_training import get_config, get_dataset, get_dataloader, get_root_model
from utils.checkpoint import load_checkpoint
from utils.transform import to_device
from utils.visualization import vis_loss_acc
from utils.tools import accuracy

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
    """针对单个样本的预测：也是最精简的一个预测流程，因为不创建数据集，不进入batch_processor.
    直接通过model得到结果，且支持cpu/GPU预测。
    注意：需要训练完成后，或在cfg中设置load_from，也就是model先加载训练好的参数文件。
    """
    pass
    
    

# %% 检测问题
from model.runner_lib import batch_detector
from utils.tools import save2pkl, loadvar
from utils.map import eval_map

def eval_dataset_det(cfg_path, 
                     load_from=None, 
                     load_device=None):
    """检测问题的eval dataset: 
    为了便于eval，不必常去修改cfg里边的设置，直接在func里边添加2个参数即可
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
    # TODO: 如下两句的顺序
    load_checkpoint(model, cfg.load_from, device)
    model = model.to(device)
    # 开始验证
    model.eval()
    all_bbox_list = []
    for c_iter, data_batch in enumerate(dataloader):
        with torch.no_grad():  # 停止反向传播，只进行前向计算
            outputs = batch_detector(model, data_batch, device, return_loss=False)
            bbox_pred = outputs['bbox']
            label_pred = outputs['label']
            all_bbox_list.append([bbox_pred, label_pred])
    # 保存预测结果到文件
    save2pkl(all_bbox_list, cfg.work_dir+'eval_result.pkl')
    # 评估
    voc_eval(all_bbox_list, dataset, iou_thr=0.5)
    
    

def predict_one_img_det():
    pass



def voc_eval(result_file, dataset, iou_thr=0.5):
    """voc数据集结果评估
    """
    if isinstance(result_file, list):
        det_results = result_file
    elif isinstance(result_file, str):
        det_results = loadvar(result_file)  # 加载结果文件
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(dataset)):   # 读取测试集的所有gt_bboxes,gt_labels
        ann = dataset.load_annotation_inds(i)
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
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
        
    eval_map(
        det_results,        # (4952,) (20,) (n,5)
        gt_bboxes,          # (4952,) (n,4)
        gt_labels,          # (4952,) (n,)
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)
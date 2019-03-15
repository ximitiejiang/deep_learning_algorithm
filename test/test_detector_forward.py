#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:27:44 2019

@author: suliang
"""
import torch

def test_ssd_forward():
    from model.one_stage_detector import OneStageDetector
    from utils.config import Config
    cfg_path = '../config/cfg_ssd300_voc.py'
    cfg = Config.fromfile(cfg_path)
    
    # prepare model
    detector = OneStageDetector(cfg, pretrained = cfg.model.pretrained)
    
    # prepare img
    img = torch.randn(2, 300, 300, 3)
    img_metas
    gt_bboxes
    gt_labels
    
    # forward
    outputs = detector(img, img_meta, return_loss=True)
    
    print(outputs.shape)

if __name__=='__main__':
    test_ssd_forward()
    
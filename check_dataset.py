#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:47:41 2019

@author: suliang
"""
from dataset.trafficsign_dataset import TrafficSign
from pre_analyze_dataset import AnalyzeDataset
from utils.config import Config
from dataset.utils import get_dataset

def main():
    config_path = './config/cfg_retinanet_r50_fpn_trafficsign_extra_aug.py'
    cfg = Config.fromfile(config_path)
    dataset = get_dataset(cfg.data.train, TrafficSign)
    ana = AnalyzeDataset('traffic_sign', dataset, checkonly=True)
    
    ana.imgcheck(1521)
    

if __name__ == "__main__":
    main()
    
    
    
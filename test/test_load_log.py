#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 08:23:17 2019

@author: ubuntu
"""

def load_log(log_path):
    """用于从log文件中载入loss数据和lr数据，两个数据可用于查看训练进展
    """
    losses = []    
    reg_losses = []
    cls_losses = []
    lrs = []
    with open(log_path) as f:
        lines = f.readlines()[4:-20]  # 头胃的额外字符需要去掉
        for line in lines:
            loss = line.split(',')[-1].strip()   # loss位置一般都是-1,
            loss_value = float(loss.split()[-1].strip())
            losses.append(loss_value)
            
            loss_reg = line.split(',')[-2].strip()   # loss位置一般都是-2,
            loss_reg_value = float(loss_reg.split()[-1].strip())
            reg_losses.append(loss_reg_value)
            
            loss_cls = line.split(',')[-3].strip()   # loss位置一般都是-3,
            loss_cls_value = float(loss_cls.split()[-1].strip())
            cls_losses.append(loss_cls_value)
            
            lr = line.split(',')[1].strip()     # lr位置一般是第1个
            lr_value = float(lr.split()[-1].strip())
            lrs.append(lr_value)

    return losses, reg_losses, cls_losses, lrs

if __name__ == "__main__":
    log_path = '../work_dirs/m2det512_voc/20190407_181009.log'  # m2det512_voc
#    log_path = '../work_dirs/ssd300_voc/20190404_181044.log'   # ssd300_voc
    losses, reg_losses, cls_losses, lrs = load_log(log_path)
    
    # draw loss curve
    import matplotlib.pyplot as plt
    import numpy as np
    plt.subplot(221)
    plt.title('loss')
    plt.plot(np.arange(len(losses)),losses)
    plt.subplot(222)
    plt.title('lr')
    plt.plot(np.arange(len(lrs)), lrs)
    plt.subplot(223)
    plt.title('loss_reg')
    plt.plot(np.arange(len(reg_losses)),reg_losses)
    plt.subplot(224)
    plt.title('loss_cls')
    plt.plot(np.arange(len(cls_losses)),cls_losses)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 08:23:17 2019

@author: ubuntu
"""

def load_log(log_path_list):
    """用于从log文件中载入loss数据和lr数据，两个数据可用于查看训练进展
    """
    assert isinstance(log_path_list, list), 'the log path should be list format.'
    losses = []    
    reg_losses = []
    cls_losses = []
    lrs = []
    epochs = []
    for i in range(len(log_path_list)):
        with open(log_path_list[i]) as f:
            lines = f.readlines()[4:]  # 头胃的额外字符需要去掉
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
                
                epoch = line.split('[')[1]
                epoch_value = float(epoch[:-1])
                epochs.append(epoch_value)

    return losses, reg_losses, cls_losses, lrs, epochs

if __name__ == "__main__":
    # m2det512
    log_path_list = ['../work_dirs/m2det512_voc/20190407_181009.log',   # 1-4
                     '../work_dirs/m2det512_voc/20190408_181146.log',   # 5-8
                     '../work_dirs/m2det512_voc/20190409_182341.log',   # 9
                     '../work_dirs/m2det512_voc/20190409_221321.log',   # 10-12
                     '../work_dirs/m2det512_voc/20190410_175628.log',   # 13-16
                     '../work_dirs/m2det512_voc/20190411_183847.log',   # 17-20
                     '../work_dirs/m2det512_voc/20190412_183426.log',   # 21
                     '../work_dirs/m2det512_voc/20190412_214256.log']   # 22-27
    # ssd300_voc_4img_per_GPU
#    log_path = ['../work_dirs/ssd300_voc/20190404_181044.log']   
    
    # ssd300_2img per GPU
#    log_path_list = ['../work_dirs/ssd300_voc/20190404_090724.log']
    
    losses, reg_losses, cls_losses, lrs, epochs = load_log(log_path_list)
    
    # draw loss curve
    import matplotlib.pyplot as plt
    import numpy as np
    plt.subplot(321)
    plt.title('loss')
    plt.plot(np.arange(len(losses)),losses)
    plt.subplot(322)
    plt.title('lr')
    plt.plot(np.arange(len(lrs)), lrs)
    plt.subplot(323)
    plt.title('loss_reg')
    plt.plot(np.arange(len(reg_losses)),reg_losses)
    plt.subplot(324)
    plt.title('loss_cls')
    plt.plot(np.arange(len(cls_losses)),cls_losses)
    plt.subplot(325)
    plt.title('epoch')
    plt.plot(np.arange(len(epochs)),epochs)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:15:27 2019

@author: ubuntu
"""


def eval_dataset_cls(cfg_path):
    """等效于runner中的val，但可用来脱离runner进行独立的数据集验证
    """
    # 准备验证所用的对象
    cfg = get_config(cfg_path)
    dataset
    dataloader
    model = get_model()
    # 开始训练
    buffer = {'acc': []}
    n_correct = 0
    model.eval()
    for c_iter, data_batch in enumerate(dataloader):
        with torch.no_grad():  # 停止反向传播，只进行前向计算
            outputs = batch_processor(self.model, data_batch, 
                                           self.loss_fn_clf,
                                           self.device,
                                           return_loss=False)
            buffer['acc'].append(outputs['acc1'])
        # 计算总体精度
        n_correct += buffer['acc'][-1] * len(data_batch['gt_labels'])
    
    vis_loss_acc(self.buffer, title='val')
    self.logger.info('ACC on valset: %.3f', self.n_correct/len(self.valset))


def eval_dataset_det():
    pass


def predict_single(self, img):
    """针对单个样本的预测：也是最精简的一个预测流程，因为不创建数据集，不进入batch_processor.
    直接通过model得到结果，且支持cpu/GPU预测。
    注意：需要训练完成后，或在cfg中设置load_from，也就是model先加载训练好的参数文件。
    """
    from utils.transformer import ImgTransform
    if self.weight_ready:
        img_transform = ImgTransform(self.cfg.transform_val)
        img, *_ = img_transform(img)
        img = img.float().to(self.device)
        # 前向计算
        y_pred = self.model(img)
        y_class = self.trainset.CLASSES[y_pred]
        self.logger.info('predict class: %s', y_class)
    else:
        raise ValueError('no model weights loaded.')
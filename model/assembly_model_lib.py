#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:22:46 2019

@author: ubuntu
"""

#from utils.module_factory import registry, build_module
import torch
from torch import nn
import torchvision
import numpy as np
from collections import OrderedDict
from utils.tools import accuracy
import torch.nn.functional as F
    
# %% onestage
class OneStageDetector(nn.Module):
    """生成单阶段的检测器：
    单阶段检测器组成：backbone + (neck) + bbox_head, 带bbox_head说明输出的就是bbox预测了
    双阶段检测器组成：backbone + (neck) + anchor_head + bbox_head, 带anchor_head说明输出的就是anchor proposal
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 创建基础模型
        from utils.prepare_training import get_model
        self.backbone = get_model(cfg.backbone)
        if cfg.neck:  # 不能写成 is not None, 因为结果是{}不是None, 但可以用True/False来判断
            self.neck = get_model(cfg.neck)
        self.bbox_head = get_model(cfg.head)  # bbox head就说明生成的是bbox
        
        # 初始化: 注意权重需要送入cpu/gpu，该步在model.to()完成
        self.init_weights()
    
    def init_weights(self):
        self.backbone.init_weights()
        if self.cfg.neck:
            self.neck.init_weights()
        self.bbox_head.init_weights()
        
        
    def forward(self, imgs, img_metas, return_loss=True, **kwargs):
        """nn.Module的forward()函数，分支成训练的forward()以及测试的forward()"""
        if return_loss:
            return self.forward_train(imgs, img_metas, **kwargs)
        else:
            return self.forward_test(imgs, img_metas, **kwargs)
        
    def forward_train(self, imgs, img_metas, **kwargs):
        """训练过程的前向计算的底层函数"""
        # 特征提取
        x = self.backbone(imgs)
        if self.cfg.neck:
            x = self.neck(x)
        outs = self.bbox_head(x)  # 获得分类和回归的预测值cls_score, bbox_preds
        # 计算损失
#        loss_inputs = outs + (gt_bboxes, gt_labels, gt_landmarks, self.cfg) # 去掉img_metas
        losses = self.bbox_head.get_losses(**outs, cfg=self.cfg, **kwargs)
        return losses
        
    def forward_test(self, imgs, img_metas, **kwargs):
        """测试过程的前向计算的底层函数: 只支持单张图片，如果多张图片则需要自定义循环
        """
        # TODO: 测试过程只需要前向计算而不需要反向传播，是否可以缩减模型尺寸?
        # 特征提取
        x = self.backbone(imgs)
        if self.cfg.neck:
            x = self.neck(x)
        outs = self.bbox_head(x)
        # 计算bbox，label
#        bbox_inputs = outs + (img_metas, self.cfg)
        bbox_results, label_results = self.bbox_head.get_bboxes(
                **outs, cfg=self.cfg, img_metas=img_metas, **kwargs)     # (k, 5), (k,)    
        
        # 把结果格式转为numpy(因为后续做mAP评估都是在cpu端numpy方式评估)
        if len(imgs) == 1:
            bbox_results = bbox_results[0]
            label_results = label_results[0]
            
            if bbox_results.shape[0] == 0:
                bbox_results = np.zeros((0, 5), dtype=np.float32)
            else:
                bbox_results = bbox_results.cpu().numpy()
                label_results = label_results.cpu().numpy()
            # 把结果格式从(k, 5)->(20,)(c,5)
            bbox_det = []
            for i in range(self.bbox_head.num_classes - 1): # 因为预测时的负样本(0)已经被筛掉了，所以这里只找20类range(20)
                inds = label_results == i + 1    # 注意：从数据集出来到训练以及预测的标签取值范围永远是1-20
                bbox_det.append(bbox_results[inds, :])
            
            return bbox_det # (n_class,)(k, 5)  不在返回labels了，因为顺序就是labels的号
        else:
            raise ValueError('currently only one batch size supported for test.')


# %%
class Segmentator(nn.Module):
    """语义分割或实例分割集成模型
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 创建基础模型
        from utils.prepare_training import get_model
        self.backbone = get_model(cfg.backbone)
        if cfg.neck:  # 不能写成 is not None, 因为结果是{}不是None, 但可以用True/False来判断
            self.neck = get_model(cfg.neck)
        self.seg_head = get_model(cfg.head)  # seg head就说明生成的是bbox
        
        # 初始化: 注意权重需要送入cpu/gpu，该步在model.to()完成
        self.init_weights()
    
    def init_weights(self):
        self.backbone.init_weights()
        if self.cfg.neck:
            self.neck.init_weights()
        self.seg_head.init_weights()
    
    def forward(self, imgs, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(imgs, **kwargs)
        else:
            return self.forward_test(imgs, **kwargs)
    
    def forward_train(self, imgs, segs, **kwargs):
        x = self.backbone(imgs)
        if self.cfg.neck:
            x = self.neck(x)
        outs = self.seg_head(x)  
        # 计算损失
        loss_inputs = [outs, segs]
        loss_dict = self.seg_head.get_losses(*loss_inputs)
        return loss_dict
    
    def forward_test(self, imgs, **kwargs):
        x = self.backbone(imgs)
        if self.cfg.neck:
            x = self.neck(x)
        outs = self.seg_head(x)  
        # TODO: 处理分割数据结果
        return outs
    
    
# %%
class Classifier(nn.Module):
    """用于分类器的总成模型"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 创建基础模型
        from utils.prepare_training import get_model
        self.backbone = get_model(cfg.backbone)
        if cfg.neck:  # 不能写成 is not None, 因为结果是{}不是None, 但可以用True/False来判断
            self.neck = get_model(cfg.neck)
        if cfg.head:
            self.cls_head = get_model(cfg.head)
        # 初始化: 注意权重需要送入cpu/gpu，该步在model.to()完成
        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if self.cfg.neck:
            self.neck.init_weights()
        if self.cfg.head:
            self.cls_head.init_weights()
    
    def forward(self, imgs, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(imgs, **kwargs)
        else:
            return self.forward_test(imgs, **kwargs)
    
    def forward_train(self, imgs, labels, **kwargs):
        preds = self.backbone(imgs)
        if self.cfg.neck:
            preds = self.neck(preds)
        if self.cfg.head:
            preds = self.cls_head(preds)  
        # 计算损失
        labels = torch.cat(labels, dim=0)  # (n,)
        loss_inputs = [preds, labels]
        out_dict = self.get_losses(*loss_inputs)
        # TODO: 增加acc计算
        acc = accuracy(preds, labels, topk=(1,5))
        out_dict.update(acc1=acc[0])
        out_dict.update(acc5=acc[1])
        return out_dict
    
    def forward_test(self, imgs, **kwargs):
        x = self.backbone(imgs)
        if self.cfg.neck:
            x = self.neck(x)
        outs = self.cls_head(x)  
        # TODO: 处理分割数据结果
        return outs
    
    def get_losses(self, preds, labels):
        """用于代替head的get_loss"""
        # TODO: 改成模块化输入参数输入
        loss_fn = F.cross_entropy
        
        losses = loss_fn(preds, labels)
        return dict(loss=losses)
    
    def get_predict(self, feat):
        """用于代替head的get_bbox"""
        pass
    

# %% two stage
class TwoStageDetector(nn.Module):
    def __init__(self):
        super().__init__()
        pass


# %%

class ExtractModel(nn.ModuleDict):
    """来自于torchvision.models._utils.IntermediateLayerGetter
    可用于从已加载权重的模型中抽取需要的层，从而不需要在load_state_dict里边取筛选相关权重名。
    会更简洁简单的生成一个新的模型，并且也不需要去检查是否有权重size不匹配的问题。
    但注意：返回的是一个字典，
    model = torch.vision.models.resnet18(pretrained=True)
    return_layers = {'layer1':'layer1', 'layer2':'layer2','layer3':'layer3'}
    model = ExtractModel(model, return_layers)
    
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    def __init__(self, model, return_layers):
        # set提取dict的keys, 确保keys都在model的子模型里边
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
            
        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        # 确保从原模型的开始部分开始提取，不能从中间提取
        # 所有相关layers都放入layers字典中
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        # 用super()的方式来封装layers，从而该对象直接就是一个ModuleDict, 该方式的优点：
        #   1. model名称就是类名
        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out  # 注意，返回的是一个字典，需要手动提取成需要的格式，比如list:  out = list(out.values())
        
if __name__ == '__main__':
    model = torchvision.models.resnet18(pretrained=True)
    return_layers = {'layer1':'layer1', 'layer2':'layer2','layer3':'layer3'}
    model = ExtractModel(model, return_layers)
    img = torch.randn((1,3,300,300))
    out = model(img)
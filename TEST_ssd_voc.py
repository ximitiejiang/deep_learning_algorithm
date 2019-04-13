"""针对faster rcnn在coco的test: 
没有用voc数据集是因为：当前虽然有faster rcnn针对voc的模型和cfg，但没有针对voc的weights
参考对标数据：mmdetection的测试结果都是基于coco_2017_train进行训练，然后基于coco_2017_val进行测试
1.
"""

import torch
import mmcv

from mmdet.datasets import build_dataloader
from mmcv.parallel import MMDataParallel
from utils.checkpoint import load_checkpoint
from dataset.utils import get_dataset
from dataset.voc_dataset import VOCDataset
from model.one_stage_detector import OneStageDetector
from utils.config import Config
from utils.map import eval_map
import numpy as np

import sys,os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)

def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def dataset_result(dataset, cfg, checkpoint_path, out_file):
    """用于指定数据集的预测结果生成：
    Args:
        dataset(obj): 数据集对象
        cfg(str):  定义cfg文件路径
        checkpoint_path(str):  定义已训练模型参数路径(模型cfg文件必须跟模型参数文件匹配)
        out_file(.pkl): 定义目标pkl文件的存放路径         
    """   
    if not out_file.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.gpus = 1
    
    if cfg.gpus == 1:
        model = OneStageDetector(cfg)
        
        load_checkpoint(model, checkpoint_path)
#        model = model.cuda()
        model = MMDataParallel(model, device_ids=[0])
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, show=False)  # (4870,)=n_imgs (20,)=n_classes (n,5)=n_bboxes
  
    if out_file:  
        print('writing results to {}'.format(out_file))  
        mmcv.dump(outputs, out_file)  # 先把模型的测试结果输出到文件中: 如果文件不存在会创建  
        

def voc_eval(result_file, dataset, iou_thr=0.5):
    """voc数据集结果评估
    """
    det_results = mmcv.load(result_file)  # 加载结果文件
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(dataset)):   # 读取测试集的所有gt_bboxes,gt_labels
        ann = dataset.get_ann_info(i)
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

if __name__ == '__main__':

    # for ssd
#    config_path = './config/cfg_ssd300_vgg16_voc.py'   # 注意：cfg和模型需要匹配，因为不同数据集类别数不一样，  
#    checkpoint_path = './weights/myssd/weight_4imgspergpu/epoch_24.pth'   
##    checkpoint_path = './work_dirs/ssd300_voc/epoch_24.pth'
##    checkpoint_path = './weights/mmdetection/ssd300_voc_vgg16_caffe_240e_20181221-2f05dd40.pth'
#    out_file = './weights/myssd/weight_4imgspergpu/results_24.pkl'

    # for m2det
    config_path = './config/cfg_m2det512_vgg16_mlfpn_voc.py'
    checkpoint_path = './weights/myssd/weight_m2det512/epoch_24.pth'
    out_file = './weights/myssd/weight_m2det512/results_24.pkl'
    
    cfg = Config.fromfile(config_path)
    dataset = get_dataset(cfg.data.test, VOCDataset)
    dataset_result(dataset, cfg, checkpoint_path, out_file)
    voc_eval(out_file, dataset, iou_thr=0.5)

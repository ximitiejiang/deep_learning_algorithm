"""针对faster rcnn在coco的test: 
没有用voc数据集是因为：当前虽然有faster rcnn针对voc的模型和cfg，但没有针对voc的weights
参考对标数据：mmdetection的测试结果都是基于coco_2017_train进行训练，然后基于coco_2017_val进行测试
1.
"""

import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors

from utils.coco_eval import evaluation

from dataset.utils import get_dataset
from dataset.voc_dataset import VOCDataset
from dataset.coco_dataset import CocoDataset
from model.one_stage_detector import OneStageDetector

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


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')  # 字符串，结果文件路径
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')  # bool变量，指明是否调用模型的show_result()显示结果
    args = parser.parse_args()
    return args


def main():
    """针对faster rcnn在voc的评估做微调
    1. args parse用直接输入替代
    2. 
    """
    config_path = './config/cfg_ssd300_vgg16_voc.py'   # 注意：cfg和模型需要匹配，因为不同数据集类别数不一样，  
    checkpoint_path = './weights/myssd/epoch_24.pth'   
    cfg = mmcv.Config.fromfile(config_path)
    out_file = 'dataset_eval_result/results.pkl'  # 注意这里要选择pkl而不能选择json，因为outputs里边包含array，用json无法保存
    eval_type = ['bbox']      # proposal_fast是mmdetection自己的实现
#    eval_type = ['proposal','bbox']   # 这几种是coco api的实现包括['proposal','bbox','segm','keypoints']，已跑通
                                    
    show_result = False   # 这里可以设置show=True从而按顺序显示每张图的测试结果(对于少量图片的数据集可以这么玩)
    
    if not out_file.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
#    cfg.model.pretrained = None

#    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    dataset = get_dataset(cfg.data.test, CocoDataset)
    
    cfg.gpus = 1
    
    if cfg.gpus == 1:
#        model = build_detector(
#            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        model = OneStageDetector(cfg)
        
        load_checkpoint(model, checkpoint_path)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, show=show_result)  
        # outputs结构: [img1,...imgn], len=5000,此为coco val的所有图片
        # 每个img结构: [cls1,...clsn], len=80, 此为每个预测的所有类的bbox预测输出
        # 每个cls结构: ndarray(n,5), 此为这个cls对应n个bbox，如果该类有预测则n>0，如果该类没有预测则n=0，第一列为置信度？
        # 注意：最内层数据结构是ndarray，是不能直接存入json文件，需要转换成data.tolist()
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(detectors, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            checkpoint_path,
            dataset,
            _data_func,
            range(cfg.gpus),
            workers_per_gpu=cfg.proc_per_gpu)
    # debug
    
    if out_file:  
        print('writing results to {}'.format(out_file))  
        mmcv.dump(outputs, out_file)  # 先把模型的测试结果输出到文件中: 如果文件不存在会创建  
        eval_types = eval_type
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = out_file
#                coco_eval(result_file, eval_types, dataset.coco)  # result_file传入coco_eval()
                """用自己写的evaluation()"""
                evaluation(result_file, dataset.coco, eval_types=eval_types)
            
            else:
                if not isinstance(outputs[0], dict):
                    result_file = out_file + '.json'
                    results2json(dataset, outputs, result_file)
#                    coco_eval(result_file, eval_types, dataset.coco)
                    """用自己写的evaluation()"""
                    evaluation(result_file, dataset.coco, eval_types=eval_types)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = out_file + '.{}.json'.format(name)
                        results2json(dataset, outputs_, result_file)
                        coco_eval(result_file, eval_types, dataset.coco)


if __name__ == '__main__':
    main()

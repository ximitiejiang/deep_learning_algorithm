#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:43:44 2019

@author: ubuntu
"""
import torch
import os
import time
import six
import pickle
from contextlib import ContextDecorator

# %%

class timer(ContextDecorator):
    """继承上下文管理装饰器实现一个双功能计时器，既可对函数计时，也可对代码块计时。
    参考：https://www.jb51.net/article/153872.htm
    继承的ContextDecorator实现了一个内部__call__()已生成了一个装饰器，内部建立上下文语义。
    所以该类是用__call__实现了装饰器计时功能，然后用上下文管理器实现代码块计时。
    
    time库的两个函数: time.time()返回的1970开始计时后的float秒数.
    time.localtime(time.time())返回struct(year,month,day,hour,min,sec,xx,xx,xx)共9个int数.
    使用方式1:
        @timer('id1')
        def fun()
    使用方式2:
        with timer('id2'):
    """
    def __init__(self, time_id):
        self.time_id = time_id
    
    def __enter__(self):        # 上下文管理器：with执行前执行
        self.start = time.time()
        
    def __exit__(self, *args):  # 上下文管理器：with执行后执行。
        elapse = time.time() - self.start
        print('timer: %s elapse %.3f seconds.'%(self.time_id, elapse))


@timer('aa')
def hello(sec):
    for i in range(5):
        time.sleep(sec)

def hello2(sec):
    with timer('position1'):
        for i in range(5):
            time.sleep(sec)
    
    
# %%

def accuracy(y_pred, label, topk=1):
    """pytorch tensor版本的精度计算：由于都是未概率化的数据，
    y_pred(b, n_classes)，tensor为未概率化的数据
    label(b,), tensor为未概率化的数据(也就是实际标签而不是独热编码)
    输出: acc (float标量)
    """
    with torch.no_grad():
        # TODO: 增加topk的功能
        if topk == 1:
            pred = torch.argmax(y_pred, dim=1)         # 输出(b,)           
            acc = (pred == label).sum().float() / len(label)
        return acc

def accuracy2(preds, targets, topk=(1, )):
    """计算指定的topk精度, topk可以是数值(1 or 5表示第1的精度或者前5的精度), 
    也可以是一组数([1,5]则表示计算第1精度和前5精度)
    args:
        preds: (b, n_class)，为
        targets: (b, ), 为实际标签(不是独热编码)
    """
    if isinstance(topk, int):
        topk = [topk]
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = preds.topk(maxk, 1, True, True) # topk(k, ?, ?, ?)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# %%    
def get_time_str():
    """计算系统时间并生成字符串"""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def exist_or_mkdir(dir_name, mode=0o777):
    """检查目录是否存在，如果不存在则创建: 可创建嵌套文件夹
    """
    dir_name = os.path.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)


# %%            
def save2pkl(var, path, to_numpy=True):
    """保存变量为.pkl文件: 输入的var变量可以是tensor，也可以是list(tensor)
    """
    if not path.endswith('.pkl') or not os.path.isdir(os.path.dirname(path)):
        raise ValueError('path format is not pkl or parent dir is not exist.')
    # 如果要转换成numpy格式
    if to_numpy:
        if isinstance(var, list) and isinstance(var[0], torch.Tensor):
            for i in range(len(var)):
                var[i] = var[i].cpu().numpy() # 如果是tensor，则先转cpu和numpy
        elif isinstance(var, torch.Tensor):
            var = var.cpu().numpy()
    # 保存
    with open(path, 'wb') as f:
        pickle.dump(var, f)


def loadvar(path):
    """采用pickle模块从pkl文件读取变量清单，但需要指定有几个变量在文件中"""
    with open(path, 'rb') as f:
        var_list = pickle.load(f)
    return var_list


# %%
from utils.visualization import vis_loss_acc
def parse_log(path, show=True):
    """解析log文件"""
    with open(path) as f:
        lines = f.readlines()
        data_dict = {'loss': [],
                     'acc': []}
        lines = lines[2:]  # 去除开始2行
        lines = lines[:-1] # 去除最后一行
        for line in lines:
            loss = float(line.split('\t')[-3].split(' ')[-1])
            acc = float(line.split('\t')[-2].split(' ')[-1])
            data_dict['loss'].append(loss)
            data_dict['acc'].append(acc)
    
    if show:
        vis_loss_acc(data_dict)
    return data_dict


# %%
from argparse import ArgumentParser

dist = dict(
        launcher='pytorch',
        backend='nccl',
        local_rank=0)


def parse_args():
    """用于从命令行获取输入参数来进行分布式训练
    使用方法： python ./train.py cfg_xxx.py  (这里省略了两个带默认值的关键字参数)
    相关参数说明：参考https://www.cnblogs.com/freshchen/p/11660046.html    
    1. 参数数据类型设置：默认的传入参数都是字符串，如果需要指定，则通过type=int来设置，此时parser会帮我们转换
    2. 参数名称设置：-n或--name代表参数的变量名，其中-n为简写名，--name为全名，一般有一个就可以，
       但如果两个都写，获取参数时需要采用全名(简写名失效)，获取变量名可以通过sys.argv[1->n]中去获取，也可以通过parser.parse_args()之后来获取。
    3. 默认参数值设置：default='Jack'，注意
    4. 参数选择范围设置：choices=[20, 30, 40]
    5. 获取参数同时触发动作设置：
        - action='store'，这是默认动作，也就是保存参数
        - action='store_true'，保存为True
        - action='append'，把值保存到列表，如果参数重复出现，则保存多个值
    """
    parser = ArgumentParser(description='Dist training argument parse')
    parser.add_argument('--task', choices=['train'])
    parser.add_argument('--config')
    parser.add_argument('--launcher', default='pytorch')   # 分布式默认启动器
    parser.add_argument('--local_rank', default=0, type=int) # 分布式默认主机
    args = parser.parse_args()       # 解析参数
    return args  # 返回已解析好的参数：使用形式为args.xxx

# %% 分布式
import torch.distributed as dist
import torch.multiprocessing as mp
def init_dist(backend='nccl', **kwargs):
    """初始化分布式系统：主要是为了启动本机多进程，一个GPU中运行一个进程
    参考：https://tramac.github.io/2019/04/22/%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83-PyTorch/
    1. 多进程启动方式：最好采用spawn
    2. 数据格式定义：需要设置属性non_blocking=True，比如input=input.cuda(non_blocking=True), 需要放在to_device()完成？
    3. batch_size：代表每个进程的batch，所以总的batch_size = batch_size * world_size
    4. workers: 表示
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')       # 多进程启动方式选择：一般有forkserver和spawn, spawn为默认方法，否则容易导致死锁
    rank = int(os.environ['RANK'])         #
    num_gpus = torch.cuda.device_count()   # 查找本机多少GPU
    torch.cuda.set_device(rank % num_gpus) # 设置
    dist.init_process_group(backend=backend, **kwargs)  # 初始化进程组


def main():
    if dist:
        world_size = torch.distributed.get_world_size()   # world_size代表
        rank = torch.distributed.get_rank()               # rank代表 
        num_workers = cfg.data_workers
        assert cfg.batch_size % world_size == 0
        batch_size = cfg.batch_size // world_size
        train_sampler = DistributedSampler(train_dataset, world_size, rank)
        val_sampler = DistributedSampler(val_dataset, world_size, rank)
        shuffle = False    
    if dist:
        model = DistributedDataParallel(
                model.cuda(), device_ids=[torch.cuda.current_device()])
    

# %%

if __name__ == "__main__":
    hello(1)
    hello2(2)
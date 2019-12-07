#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:43:44 2019

@author: ubuntu
"""
import torch
import numpy as np
import os
import time
import six
import pickle
#from contextlib import ContextDecorator
from functools import wraps

# %%
def one_hot_encode(t, n_column=None):
    """pytorch版本的独热编码生成
    args:
        t (b, ) 代表b个样本的标签，取值要从0开始，比如[0,1,2,3]
        one_hot_t (b, )
    """    
    one_hot_t = t.new_full((t.size(0), n_column), 0, dtype=torch.float32)
    inds = torch.nonzero(t >= 1).squeeze()
    if inds.numel() > 0:  # 如果有正样本则填充标签，否则自动返回0
        one_hot_t[inds, t[inds]-1] = 1  # 注意：这里要把得到的标签从[1,20]变为[0-19]，去除背景的0号标签，只计算前景。
    return one_hot_t


def label_to_onehot(labels):
    """numpy版本的标签转换为独热编码：输入的labels需要是从0开始的整数，比如[0,1,2,...]
    输出的独热编码为[[1,0,0,...],
                  [0,1,0,...],
                  [0,0,1,...]]  分别代表0/1/2的独热编码
    """
    assert labels.ndim ==1, 'labels should be 1-dim array.'
    labels = labels.astype(np.int8)
    n_col = int(np.max(labels) + 1)   # 独热编码列数，这里可以额外增加列数，填0即可，默认是最少列数
    one_hot = np.zeros((labels.shape[0], n_col))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot  # (n_samples, n_col)


def onehot_to_label(one_hot_labels):
    """把独热编码变回0-k的数字编码"""
    labels = np.argmax(one_hot_labels, axis=1)  # 提取最大值1所在列即原始从0开始的标签
    return labels


# %%
class ContextDecorator(object):
    """该类来自：from contextlib import ContextDecorator
    A base class or mixin that enables context managers to work as decorators.
    """
    def _recreate_cm(self):
        """Return a recreated instance of self.

        Allows an otherwise one-shot context manager like
        _GeneratorContextManager to support use as
        a decorator via implicit recreation.

        This is a private interface just for _GeneratorContextManager.
        See issue #11647 for details.
        """
        return self

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            with self._recreate_cm():
                return func(*args, **kwds)
        return inner
    
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


@timer('aa')                  # 使用方式1
def hello(sec):
    for i in range(5):
        time.sleep(sec)

def hello2(sec):
    with timer('position1'):  # 使用方式2
        for i in range(5):
            time.sleep(sec)
    
    
# %%

def accuracy2(preds, targets, topk=1):
    """pytorch tensor版本的精度计算：由于都是未概率化的数据，
    y_pred(b, n_classes)，tensor为未概率化的数据
    targets(b,), tensor为未概率化的数据(也就是实际标签而不是独热编码)
    输出: acc (float标量)
    """
    with torch.no_grad():
        # TODO: 增加topk的功能
        if topk == 1:
            pred = torch.argmax(preds, dim=1)         # 输出(b,)           
            acc = (pred == targets).sum().float() / len(targets)
        return acc

def accuracy(preds, targets, topk=(1, 5)):
    """计算指定的topk精度, topk可以是数值(1 or 5表示第1的精度或者前5的精度), 
    也可以是一组数([1,5]则表示计算第1精度和前5精度)
    args:
        preds: (b, n_class)，为
        targets: (b, ), 为实际标签(不是独热编码)
    """
    if isinstance(topk, int):  # topk可以是一个数值，也可以是一个list表示要求的k的个数
        topk = [topk]
    with torch.no_grad():
        maxk = max(topk)   # 提取最大k个
        batch_size = targets.size(0)

        _, pred = preds.topk(maxk, 1, True, True) # (b, 5), topk(k, ?, ?, ?)
        pred = pred.t()  # 转置(5, b)
        correct = pred.eq(targets.view(1, -1).expand_as(pred))  # (5, b)

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
def parse_log(paths, show=True):
    """解析log文件: paths代表1到多个log文件"""
    data_dict = {'loss': [],
                 'acc1': []}
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        with open(path) as f:
            lines = f.readlines()

            lines = lines[3:]  # 去除开始2行
            lines = lines[:-1] # 去除最后一行
            for line in lines:
                try:
                    loss = float(line.split('\t')[1].split(',')[0].split(' ')[-1])
                except:
                    print(line)
                try:
                    acc = float(line.split('\t')[1].split(',')[-1].split(' ')[-1])
                except:
                    print(line)
                data_dict['loss'].append(loss)
                data_dict['acc1'].append(acc)
    
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
    相关解析参数的基础说明：参考https://www.cnblogs.com/freshchen/p/11660046.html    
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
    parser.add_argument('--launcher', default='pytorch')   # 分布式默认启动器
    parser.add_argument('--local_rank', default=0, type=int) # 分布式默认主机
    args = parser.parse_args()       # 解析参数
    return args  # 返回已解析好的参数：使用形式为args.xxx

# %% 分布式
import torch.distributed as dist
import torch.multiprocessing as mp

def init_dist(backend='nccl', **kwargs):
    """初始化分布式系统：主要是为了显式地启动本机多进程，并且把训练代码拷贝给每一个进程
    参考：https://tramac.github.io/2019/04/22/%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83-PyTorch/
    并行式与分布式的区别：
    1. 并行式：
        - 只开一个进程，只适合一台机器，可以调动多块GPU，但属于假的多进程。
        - 数据分块是从batch分，每个batch复制给多个GPU，所以batch size需要设置成单卡的n倍
    - 分布式：
        - 一块GPU对应一个进程，可以一台机器，也可以多台机器，是真正的多进程。
        - 数据分块是从数据集上直接分，也就是通过DistributeSampler先把数据集分成多块给多块GPU，这避免了每个batch传输到多GPU的传输效率
          所以batch size跟单卡是一样的。
        - world_size就是
    
    分布式训练使用方法： 
    - 主机：
        python torch.distributed.launch --nproc_per_node 2      # 定义该机器的GPU数量
                                        --node_rank 0           # 定义该机器的rank等级(主机还是从机)
                                        ./train.py --config cfg_xxx.py --local_rank 0  # 定义训练文件和训练文件的子参数
    - 从机
        python torch.distributed.launch --n_proc_per_node 2 
                                        --node_rank 1
                                        ./train.py --config cfg_xxx.py --local_rank 1  #
    
    整个分布式训练过程解析：
    step0: 启动pytorch的分布式系统$python -m torch.distributed.launch train.py，用于设置环境变量，会在os.environ中创建环境变量比如RANK
        - n_proc_per_node=2表示每台机器运行的进程数
        - node_rank=0表示本节点为host主节点
        - master_addr='tcp://172.31.22.234:23456' 是为了设置主机的ip地址，是为了host主机能够被其他机器访问
        - master_port=1234 是为了设置主机的开放端口，是为了host主机能够被其他机器访问
        - train.py --config cfg.py --local_rank 0 是代表训练脚本以及训练脚本的参数
    step1: 设置多进程启动方式，mp.set_start_method('spawn')
        - 多进程启动方式：最好采用spawn
    step2: 设置本机设备，torch.cuda.set_device()
        - rank: 代表获取本机进程的优先级编号，所以叫rank，这个值取决于torch.distributed.launch中输入的--node_rank是多少，0则是主机
        - num_gpus: 代表获取本机的GPU个数
        - set_device(ran % num_gpus): 代表设置本机主设备
    step3: 初始化进程组dist.init_process_group(backend, init_method, store=None, rank=-1, world_size=-1)
        - 两种初始化进程组的方式：第一种显式定义store/rank/world_size, 第二种指定init_method为一个url地址从而所有进程都知道在哪里如何找到其他进程。    
        - backend: 表示各进程通信的底层通信协议，可以是nccl, mpi, gloo，(nccl/gloo是默认安装的，mpi必须从源码安装pytorch)建议是GPU分布训练采用nccl, cpu分布训练采用gloo
        - init_method: 表示各进程的初始化方式，可以采用dist_url，或者...
        - store:
        - rank: 
        - world_size: 总的进程数，也就是GPU数(一个GPU启动一个进程)
    step4: 创建分布式模型
        - model = DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)
    step5: 创建分布式dataloader
        - batch_size：代表每个进程的batch，所以总的batch_size = batch_size * world_size  
        - train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        - train_loader = torch.utils.data.Dataloader(trainset, 
                                                     batch_size, 
                                                     shuffle=(train_sampler is None), 
                                                     num_workers=workers, 
                                                     pin_memory=False, 
                                                     sampler=train_sampler)
    step6: 更新数据格式
        - 数据格式定义：需要设置属性non_blocking=True，比如input=input.cuda(non_blocking=True), 需要放在to_device()完成？

    4. workers: 表示在dataloader中的多线程个数，设置0表示不开多线程

    7. dist_url: 表示用来初始化进程组的方式，

    """
    # TODO: 这句话似乎不能设置spawn, 是不是直接运行mp.set_start_method('spawn')而不要if判断
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')       # 多进程启动方式选择：一般有fork和spawn, spawn为默认方法，否则容易导致死锁
    # 只有运行了torch.distributed.launch之后才能从os.environ获得到RANK变量，否则报错
    local_rank = int(os.environ['RANK'])         #
    num_gpus = torch.cuda.device_count()   # 查找本机多少GPU
    torch.cuda.set_device(local_rank % num_gpus) # 设置host主机的设备
    dist.init_process_group(backend=backend, **kwargs)  # 初始化进程组


def get_dist_info():
    """用来在dataloader中获取当前分布式的状态："""
    initialized = dist.is_initialized()
    # 如果已经初始化，即已经执行了dist.init_process_group()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    # 如果没有初始化，则当成本机一块GPU方式返回
    else:
        rank = 0
        world_size = 1
    return rank, world_size    
    



#def main():
#    if dist:
#        world_size = torch.distributed.get_world_size()   # world_size代表
#        rank = torch.distributed.get_rank()               # rank代表 
#        num_workers = cfg.data_workers
#        assert cfg.batch_size % world_size == 0
#        batch_size = cfg.batch_size // world_size
#        train_sampler = DistributedSampler(train_dataset, world_size, rank)
#        val_sampler = DistributedSampler(val_dataset, world_size, rank)
#        shuffle = False    
#    if dist:
#        model = DistributedDataParallel(
#                model.cuda(), device_ids=[torch.cuda.current_device()])
    
# %%
    



# %%

if __name__ == "__main__":
    t = torch.tensor([2, 0 , 9])
    onehot = one_hot_encode(t)
    print(onehot)
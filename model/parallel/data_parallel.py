from torch.nn.parallel import DataParallel

from .scatter_gather import scatter_kwargs


class NNDataParallel(DataParallel):
    """借用mmcv的MMDataParallel模块，改进了现有dataparallel模型的scatter函数
    1. scatter()函数：处理参数的复制问题，原有函数能处理tensor/dict/list/tuple，但由于引入datacontainer，
       所以需要对scatter函数额外增加对datacontainer的处理部分，主要是处理streams/对每个tensor转为cuda格式/复制n份给n个GPU/synchronize stream同步stream和output操作？
    2. replicas()函数：处理module的复制问题，不需要做修改
    3. parallel_apply()函数：处理threading多线程问题，即创建线程锁，创建多个线程，启动和加入多线程，多线程处理函数，这几个方面均不需要做修改
    
    """
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

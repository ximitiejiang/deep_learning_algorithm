import torch
from torch.nn.parallel._functions import Scatter as OrigScatter
from .data_container import DataContainer
from ._functions import Scatter


def scatter(inputs, target_gpus, dim=0):
    """根据输入的内容(一般是kwargs)进行分发，也就是把模型参数复制n份给每个GPU做并行计算
    该函数核心是调用递归函数scatter_map，递归得到最终每个参数的复制，多少个GPU就复制多少份
        1.如果输入是tensor则调用原有pytorch的Scatter function
        2.如果输入是DataContainer则调用自定义的Scatter function
        3.如果输入是tuple/list/dict则递归调用scatter_map()直到变为单元素不属于以上任一类型，
          然后设置scatter_map为空并返回n个单元素的list作为scatter完成的结果(即每个gpu copy一份)
    
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return OrigScatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            else:
                return Scatter.forward(target_gpus, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))]) # 一般input为空，则依然复制n份空的input
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


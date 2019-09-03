import torch
from torch.nn.parallel._functions import _get_stream


def scatter(input, devices, streams=None):
    """Scatters tensor across multiple GPUs.
    """
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [
            scatter(input[i], [devices[i // chunk_size]],   # 递归把输入的list转换为元素处理
                    [streams[i // chunk_size]]) for i in range(len(input))
        ]
        return outputs  # 最终完成数据的scatter
    elif isinstance(input, torch.Tensor):    # 递归转换之后，对数据做contiguous和cuda处理
        output = input.contiguous()
        # TODO: copy to a pinned buffer first (if copying from CPU)
        stream = streams[0] if output.numel() > 0 else None
        with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
            output = output.cuda(devices[0], non_blocking=True)
        return output
    else:
        raise Exception('Unknown type {}.'.format(type(input)))


def synchronize_stream(output, devices, streams):
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]],
                                   [streams[i]])
    elif isinstance(output, torch.Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception('Unknown type {}.'.format(type(output)))


def get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception('Unknown type {}.'.format(type(input)))


class Scatter(object):
    """用该函数来处理DataContainer的分发复制，对比pytorch自己的Scatter类：
    原来pytorch的Scatter类是一个Function类，用来处理tensor，包括了前向反向计算，所以需要用Function类
    而这个Scatter类是一个普通object类，目的只是提取DataContainer中的data数据(如果是cpu的data数据则继续递归，如果是GPU的data数据则)
    """
    @staticmethod
    def forward(target_gpus, input):
        input_device = get_input_device(input)
        streams = None
        if input_device == -1:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(device) for device in target_gpus]  

        outputs = scatter(input, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs)

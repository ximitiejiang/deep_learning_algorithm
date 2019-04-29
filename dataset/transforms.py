import mmcv
import numpy as np
import torch
import cv2

__all__ = ['ImageTransform', 'BboxTransform', 'MaskTransform', 'Numpy2Tensor']

interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4}

def imresize(img, size, return_scale=False, interpolation='bilinear'):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple): Target (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".

    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale

def imrescale(img, scale, return_scale=False, interpolation='bilinear'):
    """缩放图片：这是所有图片的强制预处理(包括train/val/test)，
    缩放比例的获得方式：min(长边要求/长边，短边要求/短边)，也就是放大比例尽可能小保证图像一定在scale方框内

    Args:
        img (ndarray): The input image.
        scale (float or tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(
                'Invalid scale {}, must be positive.'.format(scale))
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        # min(长边/长，短边/短）确保图片最大程度放大不出界
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            'Scale must be a number or tuple of int, but got {}'.format(
                type(scale)))
    # output (new_w, new_h)    
    new_size = (int(w*scale_factor + 0.5), int(h*scale_factor + 0.5))
    
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imnormalize(img, mean, std):
    img = img.astype(np.float32)  # 为避免uint8与float的计算冲突，在计算类transform都增加类型转换
    return (img - mean) / std
    

def imflip(img, flip_type='h'):
    assert flip_type in ['h','v', 'horizontal', 'vertical']
    if flip_type in ['h', 'horizontal']:
        return np.flip(img, axis=1)
    else:
        return np.flip(img, axis=0)


def impad(img, shape, pad_value=0):
    """图片扩展填充
    Args:
        img(array): img with dimension of (h,w,c)
        shape(list/tuple): size of destination size of img, (h,w) or (h,w,c)
    return:
        padded(array): padded img with dimension of (h,w,c)
    """
    if len(shape) < len(img.shape):
        shape = shape + (img.shape[-1],)
    assert len(shape)==len(img.shape)
    for i in range(len(shape) - 1):
        assert shape[i] >= img.shape[i]
    
    padded = np.empty(shape, dtype = img.dtype)
    padded[...] = pad_value
    padded[:img.shape[0], :img.shape[1], ...] = img
    return padded
    

def impad_to_multiple(img, size_divisor, pad_value=0):
    """图片扩展填充到指定倍数：
    """
    h, w, _ = img.shape
    pad_h = (1 + (h // size_divisor))*size_divisor
    pad_w = (1 + (w // size_divisor))*size_divisor
    return impad(img, (pad_h, pad_w), pad_value)


def bbox_flip2(bboxes, img_shape, flip_type='h'):
    """bbox翻转: 这是自己实现的一个
    Args:
        bboxes(list): [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
        img_shape(tuple): (h, w)
    Returns:
        fliped_img(array): (h,w,c)
    """
    assert flip_type in ['h','v', 'horizontal', 'vertical']
    bboxes=np.array(bboxes)
    h, w = img_shape[0], img_shape[1]
    assert bboxes.shape[-1] == 4
    if flip_type in ['h', 'horizontal']:
        flipped = bboxes.copy()
        # xmin = w-xmax-1, xmax = w-xmin-1
        flipped[...,0] = w - bboxes[..., 2] - 1
        flipped[...,2] = w - bboxes[..., 0] - 1
    else:
        flipped = bboxes.copy()
        flipped[...,1] = h - bboxes[..., 3] - 1
        flipped[...,3] = h - bboxes[..., 1] - 1
        
    return flipped


# %%
"""下面的相关底层函数是调用mmcv获得，也可用上面的独立函数替代
为了保证robust，暂时不替代"""
class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            # scale_factor是[ws, hs, ws, hs]
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1])
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0])
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False):
        masks = [
            mmcv.imrescale(mask, scale_factor, interpolation='nearest')
            for mask in masks
        ]
        if flip:
            masks = [mask[:, ::-1] for mask in masks]
        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])

        
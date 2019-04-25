from __future__ import division

from math import cos, pi

from .hook import Hook


class LrUpdaterHook(Hook):
    """设置warmup lr和regular lr
    流程：设置base_lr -> 设置regular_lr -> 判断epoch: 小于warmup epoch则用warmup_lr设置，大于则用regular_lr设置
    warmup lr: 在每个iter调整(before_train_iter)，通常定义在前500个iter,
    regular lr: 在每个epoch调整(before_train_epoch)，
    """
    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 **kwargs):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    '"{}" is not a supported type for warming up, valid types'
                    ' are "constant" and "linear"'.format(warmup))
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        for param_group, lr in zip(runner.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        """用该函数来计算regular lr的值，如steplr步长减缓, explr指数减缓, polylr
        被get_regular_lr调用
        """
        raise NotImplementedError

    def get_regular_lr(self, runner):
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        """用该函数来计算初始预热学习率lr"""
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in runner.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in runner.optimizer.param_groups
        ]

    def before_train_epoch(self, runner):
        """regular_lr的调整通常基于epoch，放在这里"""
        if not self.by_epoch:
            return
        self.regular_lr = self.get_regular_lr(runner)   # 更新一个list
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        """warmup的调整通常基于iter，放在这里"""
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:  # 设置warmup lr
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


class FixedLrUpdaterHook(LrUpdaterHook):
    """如果不变更学习率，则采用这个hook"""
    def __init__(self, **kwargs):
        super(FixedLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        return base_lr


class ListLrUpdaterHook(LrUpdaterHook):
    """采用list形式手动调整lr：
    比如epoch_list=[4, 12, 20], lr_list = [0.0005, 0.0002, 0.00005]
    逻辑是小于ep[i]则取lr[i]
    """
    def __init__(self, epoch_list, lr_list, **kwargs):
        super(ListLrUpdaterHook, self).__init__(**kwargs)
        self.epoch_list = epoch_list
        self.lr_list = lr_list
    
    def get_lr(self, runner, base_lr):
        progress = runner.epoch
        for ep, lr in zip(self.epoch_list, self.lr_list):
            if progress < ep:
                return lr


class StepLrUpdaterHook(LrUpdaterHook):
    """采用阶梯式学习率减缓：每个阶梯减缓gamma倍, 也就是lr*gamma^[1,2,3...]，
    如果gamma取0.1,则每个阶梯减缓1/10,1/100,...
    如果gamma取0.5,则每个接替减缓1/2,1/4...
    step指定阶梯比如[16,20]代表第16个epoch，第22个epoch, lr分别变为原来的1/10,1/100
    """
    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(StepLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        if isinstance(self.step, int):
            return base_lr * (self.gamma**(progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i       # epoch<16则exp=0, 16<epoch<22则exp=1, epoch>22则exp=2
                break
        return base_lr * self.gamma**exp   # lr*0.1^0, lr*0.1^1, lr*0.1^2


class ExpLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.gamma**progress  # lr*0.5^n


class PolyLrUpdaterHook(LrUpdaterHook):

    def __init__(self, power=1., **kwargs):
        self.power = power
        super(PolyLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        return base_lr * (1 - progress / max_progress)**self.power  # lr*((1-n)/20)^p


class InvLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * (1 + self.gamma * progress)**(-self.power)


class CosineLrUpdaterHook(LrUpdaterHook):

    def __init__(self, target_lr=0, **kwargs):
        self.target_lr = target_lr
        super(CosineLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        return self.target_lr + 0.5 * (base_lr - self.target_lr) * \
            (1 + cos(pi * (progress / max_progress)))

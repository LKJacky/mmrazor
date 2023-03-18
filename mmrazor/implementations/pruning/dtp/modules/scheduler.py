# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmrazor.registry import TASK_UTILS
from mmrazor.utils import print_log
from .mutator import BaseDTPMutator


@TASK_UTILS.register_module()
class BaseDTPScheduler:

    def __init__(
        self,
        model: nn.Module,
        mutator: BaseDTPMutator,
        flops_target=0.5,
        decay_ratio=0.6,
        refine_ratio=0.2,
        flop_loss_weight=1,
        structure_log_interval=100,
    ) -> None:
        self.model = model
        self.mutator: BaseDTPMutator = mutator
        self._init()

        self.decay_ratio = decay_ratio
        self.refine_ratio = refine_ratio
        self.flops_target = flops_target
        self.flop_loss_weight = flop_loss_weight

        self.structure_log_interval = structure_log_interval

    def _init(self):
        self.mutator.prepare_from_supernet(self.model)
        self.mutator.init_quick_flop(self.model)
        self.init_flop = self.mutator.get_soft_flop(self.model).item()
        print_log(f'Get initial flops: {self.init_flop/1e6}')

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        raise NotImplementedError()

    def after_train_forward(self, iter, epoch, max_iters, max_epochs):
        pass

    def flop_loss(self, iter, epoch, max_iters, max_epochs):
        target = self.current_target(iter, epoch, max_iters, max_epochs)
        return (self.mutator.get_soft_flop(self.model) / self.init_flop -
                target)**2

    def flop_loss_by_target(self, target):
        return (max(
            self.mutator.get_soft_flop(self.model) / self.init_flop, target) -
                target)**2

    def current_target(self, iter, epoch, max_iters, max_epochs):

        def get_target(ratio):
            assert 0 <= ratio <= 1
            return 1 - (1 - self.flops_target) * ratio

        if iter < self.decay_ratio * max_iters:
            ratio = (iter / (self.decay_ratio * max_iters))
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            ratio = 1.0
        else:
            ratio = 1.0
        return get_target(ratio)

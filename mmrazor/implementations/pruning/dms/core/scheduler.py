# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmrazor.registry import TASK_UTILS
from ...dtp.modules.dtp_adaptive import DTPAScheduler
from .mutator import DMSMutator


def to_hard(scale):
    hard = (scale >= 0.1).float()
    return hard.detach() - scale.detach() + scale


@TASK_UTILS.register_module()
class DMSScheduler(DTPAScheduler):
    mutator: DMSMutator

    def __init__(self,
                 model,
                 mutator,
                 flops_target=0.5,
                 decay_ratio=0.6,
                 refine_ratio=0.2,
                 flop_loss_weight=1,
                 by_epoch=False,
                 target_scheduler='linear',
                 structure_log_interval=100) -> None:
        super().__init__(model, mutator, flops_target, decay_ratio,
                         refine_ratio, flop_loss_weight, by_epoch,
                         structure_log_interval)
        self.target_scheduler = target_scheduler

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        self.mutator.limit_value()
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.change_hard_mode(iter, epoch, max_iters, max_epochs)
            self.mutator.ratio_train()
        else:
            self.mutator.requires_grad_(False)

    def change_hard_mode(self, iter, epoch, max_iters, max_epochs):
        if iter < (self.decay_ratio * max_iters):
            self.mutator.set_soft_flop_scale_converter(None)
        else:
            self.mutator.set_soft_flop_scale_converter(to_hard)

    def current_target(self, iter, epoch, max_iters, max_epochs):

        def get_target(ratio):
            assert 0 <= ratio <= 1
            return 1 - (1 - self.flops_target) * ratio

        if iter < self.decay_ratio * max_iters:
            if self.by_epoch:
                ratio = (epoch / (self.decay_ratio * max_epochs))
            else:
                ratio = (iter / (self.decay_ratio * max_iters))
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            ratio = 1.0
        else:
            ratio = 1.0
        if self.target_scheduler == 'linear':
            return get_target(ratio)
        elif self.target_scheduler == 'cos':
            t = get_target(1 - 0.5 * (1 + np.cos(np.pi * ratio)))
            return t
        else:
            raise NotImplementedError(f'{self.target_scheduler}')

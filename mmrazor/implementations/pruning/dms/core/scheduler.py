# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.registry import TASK_UTILS
from ...dtp.modules.dtp_adaptive import DTPAScheduler
from .mutator import DMSMutator


def to_hard(scale):
    hard = (scale >= 0.5).float()
    return hard.detach() - scale.detach() + scale


@TASK_UTILS.register_module()
class DMSScheduler(DTPAScheduler):
    mutator: DMSMutator

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

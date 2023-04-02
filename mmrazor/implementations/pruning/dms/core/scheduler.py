# Copyright (c) OpenMMLab. All rights reserved.

from mmrazor.registry import TASK_UTILS
from ...dtp.modules.dtp_adaptive import DTPAScheduler
from .mutator import DMSMutator


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
                 loss_type='l2',
                 structure_log_interval=100) -> None:
        super().__init__(model, mutator, flops_target, decay_ratio,
                         refine_ratio, flop_loss_weight, by_epoch,
                         target_scheduler, loss_type, structure_log_interval)

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        self.mutator.limit_value()
        if iter < (self.decay_ratio) * max_iters:
            self.mutator.channel_depth_train()
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.mutator.channel_train()
        else:
            self.mutator.requires_grad_(False)

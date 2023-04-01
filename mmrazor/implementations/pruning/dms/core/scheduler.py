# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

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
                 structure_log_interval=100,
                 loss_type='l2') -> None:
        super().__init__(model, mutator, flops_target, decay_ratio,
                         refine_ratio, flop_loss_weight, by_epoch,
                         structure_log_interval)
        self.target_scheduler = target_scheduler
        self.loss_type = loss_type

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        self.mutator.limit_value()
        if iter < (self.decay_ratio) * max_iters:
            self.mutator.channel_depth_train()
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.mutator.channel_train()
        else:
            self.mutator.requires_grad_(False)

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

        # get loop ratio
        def get_loop_ratio(T=1):
            if self.by_epoch:
                return (epoch % T) / T
            else:
                return (iter % T) / T

        if self.target_scheduler == 'linear':
            return get_target(ratio)
        elif self.target_scheduler == 'cos':
            t = get_target(1 - 0.5 * (1 + np.cos(np.pi * ratio)))
            return t
        elif self.target_scheduler.startswith('loop_'):
            if ratio < 1:
                T = int(self.target_scheduler[5:])
                remain_ratio = 0.5 * (1 + np.cos(np.pi * ratio))  # in [1,0]
                loop_ratio = get_loop_ratio(T)  # in [0,1]
                loop_r = 0.5 * (1 + np.cos(np.pi * loop_ratio * 2)
                                )  # 1 -> 0 -> 1
                return get_target(1 - remain_ratio * loop_r)
            else:
                return get_target(1.0)
        else:
            raise NotImplementedError(f'{self.target_scheduler}')

    def flop_loss(self, iter, epoch, max_iters, max_epochs):
        target = self.current_target(iter, epoch, max_iters, max_epochs)
        soft_flop = self.mutator.get_soft_flop(self.model) / self.init_flop

        loss_type = self.loss_type
        if loss_type == 'l2':
            loss = (soft_flop - target)**2
        elif loss_type == 'l2+':
            loss = (soft_flop - target)**2 + (soft_flop - target) * (
                1 if soft_flop > target else 0)
        elif loss_type == 'log':
            loss = torch.log(
                soft_flop / target) * (1 if soft_flop > target else 0)
        else:
            raise NotImplementedError()

        return loss

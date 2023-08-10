# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
import torch.nn as nn

from mmrazor.registry import TASK_UTILS
from mmrazor.utils import print_log
from .dtp import BaseDTPMutator
from .mutator import DMSMutator


@TASK_UTILS.register_module()
class DMSScheduler():

    # init

    def __init__(self,
                 model: nn.Module,
                 mutator: BaseDTPMutator,
                 flops_target=0.5,
                 decay_ratio=0.6,
                 refine_ratio=0.2,
                 flop_loss_weight=1,
                 by_epoch=False,
                 step=1,
                 target_scheduler='linear',
                 loss_type='l2',
                 structure_log_interval=100,
                 grad_scale=1.0,
                 train_model=True) -> None:

        self.model = model
        self.mutator: DMSMutator = mutator
        self._init()
        self.model.requires_grad_(train_model)

        self.decay_ratio = decay_ratio
        self.refine_ratio = refine_ratio
        self.flops_target = flops_target
        self.flop_loss_weight = flop_loss_weight

        self.structure_log_interval = structure_log_interval

        self.by_epoch = by_epoch

        self.target_scheduler = target_scheduler
        self.loss_type = loss_type

        if isinstance(by_epoch, bool):
            self.by_epoch = by_epoch
        elif isinstance(by_epoch, int):
            self.by_epoch = True
        else:
            raise NotImplementedError()
        self.train_model = train_model

        self.step = step

        self.grad_scale = grad_scale

    def _init(self):
        self.mutator.prepare_from_supernet(self.model)
        self.mutator.init_quick_flop(self.model)
        self.init_flop = self.mutator.get_soft_flop(self.model).item()
        print_log(f'Get initial soft flops: {self.init_flop/1e6}')

    # hook

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        self.mutator.limit_value()
        if iter < (self.decay_ratio) * max_iters:
            self.mutator.channel_depth_train()
            self.model.requires_grad_(self.train_model)
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.mutator.channel_train()
            self.model.requires_grad_(self.train_model)
        else:
            self.mutator.requires_grad_(False)

    def after_train_forward(self, iter, epoch, max_iters, max_epochs):
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            res = {
                'flops_loss':
                self.flop_loss(iter, epoch, max_iters, max_epochs) *
                self.flop_loss_weight,
                'soft_flop':
                self.mutator.get_soft_flop(self.model).detach() / 1e6,
                'target':
                self.current_target(iter, epoch, max_iters, max_epochs)
            }
            return res
        else:
            return {}

    # flops

    def current_target(self, iter, epoch, max_iters, max_epochs):
        # epoch = epoch + 1

        def get_target(ratio):
            assert 0 <= ratio <= 1
            return 1 - (1 - self.flops_target) * ratio

        if iter < self.decay_ratio * max_iters:
            if self.by_epoch:
                ratio = (
                    epoch // self.step * self.step /
                    (self.decay_ratio * max_epochs))
            else:
                ratio = (
                    iter // self.step * self.step /
                    (self.decay_ratio * max_iters))
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
        elif self.target_scheduler == 'root':
            t = self.flops_target**(ratio)
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
            loss = (soft_flop - target)**2 * (1 if soft_flop > target else 0)
        elif loss_type == 'l2+':
            loss = (soft_flop - target)**2 + (soft_flop - target) * (
                1 if soft_flop > target else 0)
        elif loss_type == 'log':
            loss = torch.log(
                soft_flop / target) * (1 if soft_flop > target else 0)
        else:
            raise NotImplementedError()

        return loss

    def flop_loss_by_target(self, target):
        return (max(
            self.mutator.get_soft_flop(self.model) / self.init_flop, target) -
                target)**2

    def norm_grad(self):
        self.mutator.norm_gradient(self.grad_scale)

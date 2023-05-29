# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import numpy as np
import torch
import torch.nn as nn

from mmrazor.registry import MODELS, TASK_UTILS
from .mutable_channels import BaseDTPMutableChannel, dtp_get_importance
from .mutator import BaseDTPMutator
# dtp with feature norm
from .scheduler import BaseDTPScheduler
from .unit import BaseDTPUnit

# dtp with adaptive importance


class DTPAMutableChannelImp(BaseDTPMutableChannel):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)
        self.imp = nn.parameter.Parameter(torch.zeros([num_channels]))
        self.requires_grad_(False)

    @property
    def current_imp(self):
        if self.imp.min() == self.imp.max():
            return self.imp.new_ones(
                [self.num_channels]).detach() - self.imp.detach() + self.imp
        else:
            w = dtp_get_importance(self.imp, self.e)
            if self.training:
                with torch.no_grad():
                    self.mask.data = (w >= 0.5).float()
            return w

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.num_channels, 1.0)


@MODELS.register_module()
class DTPAUnit(BaseDTPUnit):

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__(num_channels)
        self.mutable_channel: DTPAMutableChannelImp = DTPAMutableChannelImp(
            self.num_channels)
        self.requires_grad_(False)

    @torch.no_grad()
    def info(self) -> str:
        return (f'{self.mutable_channel.imp.min().item():.3f}\t'
                f'{self.mutable_channel.imp.max().item():.3f}\t'
                f'{self.mutable_channel.e.item():.3f}')  # noqa


class DTPAMutator(BaseDTPMutator):

    def __init__(
        self,
        channel_unit_cfg=dict(type='DTPAUnit', default_args=dict()),
        parse_cfg=dict(
            _scope_='mmrazor',
            type='ChannelAnalyzer',
            demo_input=dict(
                type='DefaultDemoInput',
                input_shape=[1, 3, 224, 224],
            ),
            tracer_type='FxTracer',
        )
    ) -> None:
        super().__init__(channel_unit_cfg, parse_cfg)
        self.mutable_units: List[DTPAUnit]

    @torch.no_grad()
    def limit_value(self):
        for unit in self.mutable_units:
            unit.mutable_channel.limit_value()

    def ratio_train(self):
        for unit in self.mutable_units:
            unit.requires_grad_(True)


@TASK_UTILS.register_module()
class DTPAScheduler(BaseDTPScheduler):
    mutator: DTPAMutator

    def __init__(self,
                 model: nn.Module,
                 mutator: BaseDTPMutator,
                 flops_target=0.5,
                 decay_ratio=0.6,
                 refine_ratio=0.2,
                 flop_loss_weight=1,
                 by_epoch=False,
                 target_scheduler='linear',
                 loss_type='l2',
                 structure_log_interval=100) -> None:
        super().__init__(model, mutator, flops_target, decay_ratio,
                         refine_ratio, flop_loss_weight,
                         structure_log_interval)
        self.by_epoch = by_epoch

        self.target_scheduler = target_scheduler
        self.loss_type = loss_type

        if isinstance(by_epoch, bool):
            self.by_epoch = by_epoch
            self.epoch_T = 1
        elif isinstance(by_epoch, int):
            self.by_epoch = True
            self.epoch_T = by_epoch
        else:
            raise NotImplementedError()

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        self.mutator.limit_value()
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.mutator.ratio_train()
        else:
            self.mutator.requires_grad_(False)

    def after_train_forward(self, iter, epoch, max_iters, max_epochs):
        super().after_train_forward(iter, epoch, max_iters, max_epochs)
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            res = {
                'flops_loss':
                self.flop_loss(iter, epoch, max_iters, max_epochs) *
                self.flop_loss_weight,
                'soft_flop':
                self.mutator.get_soft_flop(self.model).detach(),
                'target':
                self.current_target(iter, epoch, max_iters, max_epochs)
            }
            return res
        else:
            return {}

    def current_target(self, iter, epoch, max_iters, max_epochs):

        def get_target(ratio):
            assert 0 <= ratio <= 1
            return 1 - (1 - self.flops_target) * ratio

        if iter < self.decay_ratio * max_iters:
            if self.by_epoch:
                ratio = (
                    epoch // self.epoch_T * self.epoch_T /
                    (self.decay_ratio * max_epochs))
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

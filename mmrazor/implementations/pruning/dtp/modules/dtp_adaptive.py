# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

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


@MODELS.register_module()
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

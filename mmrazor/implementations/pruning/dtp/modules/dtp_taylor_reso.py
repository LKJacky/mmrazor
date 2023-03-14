# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import List

import torch

from mmrazor.registry import MODELS, TASK_UTILS
from .dtp_adaptive import DTPAMutator
from .dtp_taylor import (DTPTMutableChannelImp, DTPTUnit,
                         taylor_backward_hook_wrapper)
from .scheduler import BaseDTPScheduler

# dtp with taylor importance base dtp with adaptive importance


@torch.jit.script
def dtopk_reso(x: torch.Tensor, e: torch.Tensor, resolution=1 / 1000):
    # add min or max
    # e = soft_clip(e, 1 / x.numel(), 1)
    y: torch.Tensor = -(x - e) * (4 / (resolution + 1e-8))
    s = y.sigmoid()
    return s


@torch.jit.script
def dtp_get_importance(v: torch.Tensor, e: torch.Tensor, resolution=1 / 1000):
    vm = v.unsqueeze(-1) - v.unsqueeze(0)
    vm = (vm >= 0).float() - vm.detach() + vm
    v_union = vm.mean(dim=-1)  # big to small
    return dtopk_reso(1 - v_union, e, resolution=resolution)


class DTPTRMutableChannelImp(DTPTMutableChannelImp):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.reso = 1

    @property
    def current_imp(self):
        e_imp = dtp_get_importance(self.taylor, self.e, resolution=self.reso)
        if self.training and e_imp.requires_grad:
            e_imp.register_hook(taylor_backward_hook_wrapper(self, e_imp))
        if self.training:
            with torch.no_grad():
                self.mask.data = (e_imp >= 0.5).float()
        return e_imp


@MODELS.register_module()
class DTPTRUnit(DTPTUnit):

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__(num_channels)
        self.mutable_channel: DTPTRMutableChannelImp = DTPTRMutableChannelImp(
            self.num_channels)
        self.requires_grad_(False)


@MODELS.register_module()
class DTPTRMutator(DTPAMutator):

    def __init__(
        self,
        channel_unit_cfg=dict(type='DTPTRUnit', default_args=dict()),
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
        self.mutable_units: List[DTPTRUnit]

    def set_reso(self, reso=1.0):
        for unit in self.mutable_units:
            unit.mutable_channel.reso = reso


@TASK_UTILS.register_module()
class DTPTRScheduler(BaseDTPScheduler):
    mutator: DTPTRMutator

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        self.mutator.limit_value()
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.mutator.ratio_train()
        else:
            self.mutator.requires_grad_(False)
        self.mutator.set_reso(
            self.current_reso(iter, epoch, max_iters, max_epochs))

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
                self.current_target(iter, epoch, max_iters, max_epochs),
                'reso':
                self.current_reso(iter, epoch, max_iters, max_epochs)
            }
            return res
        else:
            return {}

    def current_reso(self, iter, epoch, max_iters, max_epochs):

        def get_reso(ratio):
            return (math.cos(ratio * math.pi) + 1) / 2

        if iter < self.decay_ratio * max_iters:
            ratio = (iter / (self.decay_ratio * max_iters))
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            ratio = 1.0
        else:
            ratio = 1.0
        return get_reso(ratio)

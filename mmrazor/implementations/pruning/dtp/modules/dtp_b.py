# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import List

import torch
import torch.nn as nn

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.mutables import L1MutableChannelUnit
from mmrazor.registry import MODELS
from mmrazor.utils import print_log
from .mutable_channels import (BaseDTPMutableChannel,
                               ImpMutableChannelContainer, dtopk)
from .mutator import BaseDTPMutator
from .ops import ImpBatchnorm2d, ImpConv2d, ImpLinear


class DTPBMutableChannelImp(BaseDTPMutableChannel):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.v = nn.parameter.Parameter(
            torch.tensor([0.0] * self.num_channels))
        self.e = nn.parameter.Parameter(torch.tensor([1.0]))
        self.requires_grad_(False)

        self.per_channel_mode = False

    def get_importance(self, v: torch.Tensor, e: torch.Tensor):
        if self.per_channel_mode is False:
            vm = v.unsqueeze(-1) - v.unsqueeze(0)
            vm = (vm >= 0).float() - vm.detach() + vm
            v = vm.mean(dim=-1)
            return dtopk(v, e)
        else:
            mask = ((self.v - self.e) >= 0.0).float()
            return mask.detach() - self.v.detach() + self.v

    @property
    def current_imp(self):
        if self.e.requires_grad is False and self.v.requires_grad is False:
            return self.mask
        else:
            dis = self.v.max() - self.v.min()
            if dis.abs() == 0.0:
                return self.v.new_ones(
                    self.v.shape) - self.v.detach() + self.v + 0 * self.e
            else:
                imp = self.get_importance(self.v, self.e)
                with torch.no_grad():
                    self.mask.data = (imp >= 0.5).float()
                return imp

    @property
    def current_imp_flop(self):
        dis = self.v.max() - self.v.min()
        if dis == 0:
            return self.v.new_ones(
                self.v.shape) - self.v.detach() + self.v + 0 * self.e
        else:
            imp = self.get_importance(self.v, self.e)
            return imp

    @torch.no_grad()
    def limit_value(self):
        if not self.per_channel_mode:
            self.e.data = torch.clamp(self.e, 1 / self.num_channels, 1.0)
        pass

    @torch.no_grad()
    def mutate_param_per_channel(self):
        self.per_channel_mode = False
        imp = self.get_importance(self.v, self.e)
        mask = imp >= 0.5
        self.v.data[mask] = 0.05
        self.v.data[~mask] = -0.05
        self.e.data.fill_(0.0)
        self.per_channel_mode = True

    # train mode


@MODELS.register_module()
class DTPBUnit(L1MutableChannelUnit):

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__(num_channels, choice_mode='number')
        self.mutable_channel: DTPBMutableChannelImp = DTPBMutableChannelImp(
            self.num_channels)
        self.requires_grad_(False)

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: ImpConv2d,
                nn.BatchNorm2d: ImpBatchnorm2d,
                nn.Linear: ImpLinear,
                nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
            })
        self._register_channel_container(model, ImpMutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    def info(self) -> str:
        return f'{self.mutable_channel.v.min().item():.3f}\t{self.mutable_channel.v.max().item():.3f}\t{self.mutable_channel.e.item():.3f}'  # noqa

    def per_channel_train(self):
        self.requires_grad_(False)
        self.mutable_channel.v.requires_grad_(True)
        self.mutable_channel.per_channel_mode = True

    def ratio_train(self):
        self.mutable_channel.requires_grad_(True)


@MODELS.register_module()
class DTPBMutator(BaseDTPMutator):

    def __init__(
        self,
        channel_unit_cfg=dict(type='DTPBUnit', default_args=dict()),
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
        self.mutable_units: List[DTPBUnit]

    def limit_value(self):
        for unit in self.mutable_units:
            unit.mutable_channel.limit_value()

    def to_per_channel_train(self):
        print_log('To per channel mode')
        for unit in self.mutable_units:
            unit.mutable_channel.mutate_param_per_channel()

    def ratio_train(self):
        for unit in self.mutable_units:
            unit.ratio_train()

    def per_channel_train(self):
        for unit in self.mutable_units:
            unit.per_channel_train()

    def info(self):
        import json
        res = ''
        structure = self.current_choices
        res += (json.dumps(structure, indent=4)) + '\n'
        for unit in self.mutable_units:
            res += (f'{unit.name}:\t{unit.info()}') + '\n'
        return res


class DTPBScheduler:

    def __init__(
        self,
        model: nn.Module,
        mutator: DTPBMutator,
        flops_target=0.5,
        decay_ratio=0.6,
        refine_ratio=0.2,
    ) -> None:
        self.model = model
        self.mutator: DTPBMutator = mutator
        self.mutator.init_quick_flop(self.model)
        self.init_flop = self.mutator.get_soft_flop(self.model).item()
        print(f'Get initial flops: {self.init_flop/1e6}')

        self.decay_ratio = decay_ratio
        self.refine_ratio = refine_ratio
        self.flops_target = flops_target

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        if iter < self.decay_ratio * max_iters:
            self.mutator.ratio_train()
        elif iter == math.ceil(self.decay_ratio * max_iters):
            self.mutator.to_per_channel_train()
            self.mutator.per_channel_train()
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.mutator.per_channel_train()
        else:
            self.mutator.requires_grad_(False)
        self.mutator.limit_value()

    def after_train_forward(self, iter, epoch, max_iters, max_epochs):
        if iter < self.decay_ratio * max_iters:
            return {
                'flops_loss': self.flop_loss(iter, epoch, max_iters,
                                             max_epochs),
                'soft_flop': self.mutator.get_soft_flop(self.model).detach(),
                'target': self.current_target(iter, epoch, max_iters,
                                              max_epochs)
            }
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            return {
                'flops_loss': self.flop_loss(iter, epoch, max_iters,
                                             max_epochs),
                'soft_flop': self.mutator.get_soft_flop(self.model).detach(),
                'target': self.current_target(iter, epoch, max_iters,
                                              max_epochs)
            }
        else:
            return {}

    def flop_loss(self, iter, epoch, max_iters, max_epochs):
        target = self.current_target(iter, epoch, max_iters, max_epochs)
        return (self.mutator.get_soft_flop(self.model) / self.init_flop -
                target)**2

    def current_target(self, iter, epoch, max_iters, max_epochs):
        if iter < self.decay_ratio * max_iters:
            return 1 - (1 - self.flops_target) * (
                iter / (self.decay_ratio * max_iters))
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            return self.flops_target
        else:
            return self.flops_target

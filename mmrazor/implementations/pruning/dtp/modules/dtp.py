# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch
import torch.nn as nn

from mmrazor.registry import MODELS, TASK_UTILS
from mmrazor.utils import print_log
from .mutable_channels import BaseDTPMutableChannel, dtopk
from .mutator import BaseDTPMutator
from .scheduler import BaseDTPScheduler
from .unit import BaseDTPUnit


class DTPMutableChannelImp(BaseDTPMutableChannel):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)

        index = torch.linspace(0, 1, num_channels)
        self.register_buffer('index', index)
        self.index: torch.Tensor
        self.lamda = 1.0

    @property
    def current_imp(self):
        w = dtopk(self.index, self.e, self.lamda)
        with torch.no_grad():
            self.mask.data = (w >= 0.5).float()
        return w

    @property
    def current_imp_flop(self):
        return self.current_imp

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.num_channels, 1.0)


@MODELS.register_module()
class DTPUnit(BaseDTPUnit):

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__(num_channels)
        self.mutable_channel: DTPMutableChannelImp = DTPMutableChannelImp(
            self.num_channels)
        self.requires_grad_(False)

    def info(self) -> str:
        return f'{self.mutable_channel.e.item():.3f}'  # noqa

    @torch.no_grad()
    def resort(self):
        norm = self._get_unit_norm()
        imp = norm

        index = imp.sort(descending=True)[1]  # index of big to small
        index_space = torch.linspace(
            0, 1, self.num_channels, device=index.device)  # 0 -> 1
        new_index = torch.zeros_like(imp).scatter(0, index, index_space)
        self.mutable_channel.index.data = new_index


@MODELS.register_module()
class DTPMutator(BaseDTPMutator):

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
        self.mutable_units: List[DTPUnit]

    def limit_value(self):
        for unit in self.mutable_units:
            unit.mutable_channel.limit_value()

    def ratio_train(self):
        for unit in self.mutable_units:
            unit.requires_grad_(True)

    def resort(self):
        print_log('Resort')
        for unit in self.mutable_units:
            unit.resort()


@TASK_UTILS.register_module()
class DTPScheduler(BaseDTPScheduler):
    mutator: DTPMutator

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        if iter == 0:
            self.mutator.resort()
            self.mutator.ratio_train()
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.mutator.ratio_train()
        else:
            self.mutator.requires_grad_(False)
        self.mutator.limit_value()

    def after_train_forward(self, iter, epoch, max_iters, max_epochs):
        super().after_train_forward(iter, epoch, max_iters, max_epochs)
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            return {
                'flops_loss':
                self.flop_loss(iter, epoch, max_iters, max_epochs) *
                self.flop_loss_weight,
                'soft_flop':
                self.mutator.get_soft_flop(self.model).detach(),
                'target':
                self.current_target(iter, epoch, max_iters, max_epochs)
            }
        else:
            return {}

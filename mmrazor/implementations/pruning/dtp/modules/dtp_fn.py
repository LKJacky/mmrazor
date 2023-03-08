# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch
import torch.nn as nn
from torch import distributed as dist

from mmrazor.registry import MODELS, TASK_UTILS
from mmrazor.utils import print_log
from .mutable_channels import BaseDTPMutableChannel, dtopk
from .mutator import BaseDTPMutator
# dtp with feature norm
from .ops import CollectMixin
from .scheduler import BaseDTPScheduler
from .unit import BaseDTPUnit


class DTPFNMutableChannelImp(BaseDTPMutableChannel):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)

        imp = torch.zeros([num_channels])
        self.register_buffer('imp', imp)
        self.imp: torch.Tensor

    def get_importance(self, v: torch.Tensor, e: torch.Tensor):
        vm = v.unsqueeze(-1) - v.unsqueeze(0)
        vm = (vm >= 0).float() - vm.detach() + vm
        v_union = vm.mean(dim=-1)  # big to small
        return dtopk(1 - v_union, e)

    @property
    def current_imp(self):
        if self.imp.min() == self.imp.max():
            return self.imp.new_ones([self.num_channels])
        w = self.get_importance(self.imp, self.e)
        with torch.no_grad():
            self.mask.data = (w >= 0.5).float()
        return w

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.num_channels, 1.0)


@MODELS.register_module()
class DTPFNUnit(BaseDTPUnit):

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__(num_channels)
        self.mutable_channel: DTPFNMutableChannelImp = DTPFNMutableChannelImp(
            self.num_channels)
        self.requires_grad_(False)
        self.decay = 0.99

    @torch.no_grad()
    def info(self) -> str:
        return (f'{self.mutable_channel.imp.min().item():.3f}\t'
                f'{self.mutable_channel.imp.max().item():.3f}\t'
                f'{self.mutable_channel.e.item():.3f}')  # noqa

    @torch.no_grad()
    def update_imp(self):
        new_imp = None
        for module in self.input_related_collect_ops:
            module: CollectMixin
            assert len(module.recorded_input) == 1
            input: torch.Tensor = module.recorded_input[0]
            if len(input.shape) == 2:
                norm = self.get_feature_norm(input.unsqueeze(-1))
            elif len(input.shape) == 4:
                norm = self.get_feature_norm(input.flatten(2))
            else:
                raise NotImplementedError()
            if new_imp is None:
                new_imp = norm
            else:
                new_imp[new_imp < norm] = norm[new_imp < norm]
        assert isinstance(new_imp, torch.Tensor)
        if dist.is_initialized():
            dist.all_reduce(new_imp)
        self.mutable_channel.imp = self.decay * self.mutable_channel.imp + (
            1 - self.decay) * new_imp

    def get_feature_norm(self, feature: torch.Tensor):
        assert len(feature.shape) == 3  # B C N
        return feature.norm(p=1, dim=-1).mean(dim=0)


@MODELS.register_module()
class DTPFNMutator(BaseDTPMutator):

    def __init__(
        self,
        channel_unit_cfg=dict(type='DTPFNUnit', default_args=dict()),
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
        self.mutable_units: List[DTPFNUnit]

    @torch.no_grad()
    def limit_value(self):
        for unit in self.mutable_units:
            unit.mutable_channel.limit_value()

    def ratio_train(self):
        for unit in self.mutable_units:
            unit.requires_grad_(True)

    @torch.no_grad()
    def update_imp(self) -> None:
        """Start recording the related information."""
        for unit in self.mutable_units:
            unit.update_imp()


@TASK_UTILS.register_module()
class DTPFNScheduler(BaseDTPScheduler):
    mutator: DTPFNMutator

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        self.mutator.limit_value()
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            try:
                self.mutator.update_imp()
            except Exception as e:
                print_log(f'update imp error, as {e}')
            self.mutator.reset_recorded_info()
            self.mutator.ratio_train()
            self.mutator.start_record_info()
        else:
            self.mutator.requires_grad_(False)

    def after_train_forward(self, iter, epoch, max_iters, max_epochs):
        super().after_train_forward(iter, epoch, max_iters, max_epochs)
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.mutator.end_record_info()
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

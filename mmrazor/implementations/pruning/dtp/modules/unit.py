# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.mutables import L1MutableChannelUnit
from mmrazor.registry import MODELS
from ...chip.collect.unit import CollectUnitMixin
from .mutable_channels import (DTPAdaptiveMutableChannelImp,
                               DTPMutableChannelImp,
                               ImpMutableChannelContainer, PASMutableChannel,
                               grad_adjust_wrapper)
from .ops import ImpBatchnorm2d, ImpConv2d, ImpLinear, ImpModuleMixin


class BaseDTPUnit(L1MutableChannelUnit, CollectUnitMixin):

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__(num_channels, choice_mode='number')
        self.mutable_channel: DTPMutableChannelImp = DTPMutableChannelImp(
            self.num_channels)
        self.requires_grad_(False)

    def prepare_for_pruning(self, model: nn.Module):

        from mmrazor.implementations.pruning.dms.core.models.swin import (
            BaseShiftedWindowAttention, ImpShiftedWindowAttention)
        from .ops import ImpLayerNorm
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: ImpConv2d,
                nn.BatchNorm2d: ImpBatchnorm2d,
                nn.Linear: ImpLinear,
                nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                BaseShiftedWindowAttention: ImpShiftedWindowAttention,
                nn.LayerNorm: ImpLayerNorm,
            })
        self._register_channel_container(model, ImpMutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    def info(self):
        raise NotImplementedError()


@MODELS.register_module()
class ImpUnit(L1MutableChannelUnit):

    def __init__(
        self,
        num_channels: int,
        imp_type='dtp',
        grad_clip=-1,
        grad_mode=None,
        index_revert=False,
        ste=False,
    ) -> None:
        super().__init__(num_channels, choice_mode='number')
        delta_limit = grad_clip

        assert imp_type in ['dtp', 'pas', 'dtp_a']
        self.imp_type = imp_type

        self.mutable_channel: Union[PASMutableChannel, DTPMutableChannelImp]
        if self.imp_type == 'pas':
            self.mutable_channel = PASMutableChannel(self.num_channels)
        elif self.imp_type == 'dtp':
            self.mutable_channel = DTPMutableChannelImp(  # noqa
                self.num_channels, delta_limit=delta_limit)
        elif self.imp_type == 'dtp_a':
            self.mutable_channel = DTPAdaptiveMutableChannelImp(  # noqa
                self.num_channels, delta_limit=delta_limit)
        else:
            raise NotImplementedError(self.imp_type)
        self.requires_grad_(False)

        self.index_revert = index_revert

        self.grad_mode = grad_mode
        self.ste = ste

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
        for channel in self.input_related + self.output_related:
            if isinstance(channel.module, ImpModuleMixin):
                channel.module.ste = self.ste

    @torch.no_grad()
    def resort(self):
        if isinstance(self.mutable_channel, DTPMutableChannelImp):
            norm = self._get_unit_norm()
            imp = norm

            index = imp.sort(
                descending=(not self.index_revert))[1]  # index of big to small
            index_space = torch.linspace(
                0, 1, self.num_channels, device=index.device)  # 0 -> 1
            new_index = torch.zeros_like(imp).scatter(0, index, index_space)
            self.mutable_channel.index.data = new_index

    @torch.no_grad()
    def importance(self):
        if self.imp_type == 'dtp':
            return self.mutable_channel.current_imp
        elif self.imp_type == 'pas':
            return self.mutable_channel.imp.detach()
        elif self.imp_type == 'dtp_a':
            return self.mutable_channel.v.detach()
        else:
            raise NotImplementedError()

    def activate_grad(self):
        self.requires_grad_(True)
        if isinstance(self.mutable_channel,
                      DTPMutableChannelImp) or isinstance(
                          self.mutable_channel, DTPAdaptiveMutableChannelImp):
            self.mutable_channel.e.register_hook(
                grad_adjust_wrapper(self.grad_mode))

    def info(self) -> str:
        if self.imp_type == 'dtp_a':
            return f'imp: {self.mutable_channel.v.min().item():.3f}\t{self.mutable_channel.v.max().item():.3f}\t{self.mutable_channel.e.item():.3f}'  # noqa
        elif self.imp_type == 'pas':
            return f'imp: {self.mutable_channel.imp.min().item():.3f}\t{self.mutable_channel.imp.max().item():.3f}'  # noqa
        else:
            raise NotImplementedError()

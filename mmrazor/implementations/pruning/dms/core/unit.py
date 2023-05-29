# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.mutables import L1MutableChannelUnit
from mmrazor.registry import MODELS
from ...chip.collect.unit import CollectUnitMixin
from ...dtp.modules.dtp_taylor import DMSMutableMixIn
from ...dtp.modules.mutable_channels import (DTPMutableChannelImp,
                                             ImpMutableChannelContainer,
                                             SimpleMutableChannel)
from ...dtp.modules.ops import ImpBatchnorm2d, ImpConv2d, ImpLinear


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
        from ...dtp.modules.ops import ImpLayerNorm
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


class DTPTMutableChannelImp(SimpleMutableChannel, DMSMutableMixIn):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self._dms_mutable_mixin_init(self.num_channels)

    def fix_chosen(self, chosen=None):
        return super().fix_chosen(chosen)


@MODELS.register_module()
class DTPTUnit(BaseDTPUnit):

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__(num_channels)
        self.mutable_channel: DTPTMutableChannelImp = DTPTMutableChannelImp(
            self.num_channels)
        self.requires_grad_(False)

    @torch.no_grad()
    def info(self) -> str:
        return (f'taylor: {self.mutable_channel.taylor.min().item():.3f}\t'
                f'{self.mutable_channel.taylor.max().item():.3f}\t'
                f'e: {self.mutable_channel.e.item():.3f}')  # noqa

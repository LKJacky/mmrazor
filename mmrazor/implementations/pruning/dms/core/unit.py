# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.mutables import L1MutableChannelUnit
from mmrazor.registry import MODELS
from ...chip.collect.unit import CollectUnitMixin
from .models.opt.opt_ops import (ImpEmbedding, ImpOPTAttention,
                                 ImpOPTLearnedPositionalEmbedding,
                                 OPTAttention, OPTLearnedPositionalEmbedding)
from .mutable import DTPTMutableChannelImp, ImpMutableChannelContainer
from .op import ImpBatchnorm2d, ImpConv2d, ImpLinear

DTPMutableChannelImp = DTPTMutableChannelImp


class BaseDTPUnit(L1MutableChannelUnit, CollectUnitMixin):

    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels, choice_mode='number')
        self.mutable_channel: DTPMutableChannelImp = DTPMutableChannelImp(
            self.num_channels)
        self.requires_grad_(False)
        from mmrazor.implementations.pruning.dms.core.models.swin import (
            BaseShiftedWindowAttention, ImpShiftedWindowAttention)
        from ...dtp.modules.ops import ImpLayerNorm
        self.module_mapping = {
            nn.Conv2d: ImpConv2d,
            nn.BatchNorm2d: ImpBatchnorm2d,
            nn.Linear: ImpLinear,
            nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
            BaseShiftedWindowAttention: ImpShiftedWindowAttention,
            nn.LayerNorm: ImpLayerNorm,
            nn.Embedding: ImpEmbedding,
            OPTLearnedPositionalEmbedding: ImpOPTLearnedPositionalEmbedding,
            OPTAttention: ImpOPTAttention,
        }

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(model, self.module_mapping)
        self._register_channel_container(model, ImpMutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    def info(self):
        raise NotImplementedError()


@MODELS.register_module()
class DTPTUnit(BaseDTPUnit):

    def __init__(
        self,
        num_channels: int,
        extra_mapping={},
    ) -> None:
        super().__init__(num_channels)
        self.mutable_channel: DTPTMutableChannelImp = DTPTMutableChannelImp(
            self.num_channels)
        self.requires_grad_(False)
        self.module_mapping.update(extra_mapping)

    @torch.no_grad()
    def info(self) -> str:
        return (f'taylor: {self.mutable_channel.info()}\t')  # noqa

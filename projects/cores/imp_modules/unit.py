import torch.nn as nn

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.mutables import L1MutableChannelUnit
from .ops import (ImpBatchnorm2d, ImpConv2d, ImpLinear,
                  ImpMutableChannelContainer, SimpleMutableChannelImp)


class ImpUnit(L1MutableChannelUnit):

    def __init__(self,
                 num_channels: int,
                 choice_mode='number',
                 divisor=1,
                 min_value=1,
                 min_ratio=0.9) -> None:
        super().__init__(num_channels, choice_mode, divisor, min_value,
                         min_ratio)
        self.mutable_channel: SimpleMutableChannelImp = SimpleMutableChannelImp(  # noqa
            self.num_channels)

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

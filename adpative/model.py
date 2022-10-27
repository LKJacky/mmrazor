from typing import Union

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

import mmrazor.models.architectures.dynamic_ops as dynamic_ops
from mmrazor.models.architectures import DynamicConv2d
from mmrazor.models.mutables.mutable_channel import (L1MutableChannelUnit,
                                                     MutableChannelContainer,
                                                     SimpleMutableChannel)
from mmrazor.registry import MODELS


@MODELS.register_module()
class MutableImportance(SimpleMutableChannel):

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.importance = nn.parameter.Parameter(torch.ones([num_channels]))
        self.importance.requires_grad_()


@MODELS.register_module()
class MutableImpChannelContainer(MutableChannelContainer):

    @property
    def imp(self):
        imp = []
        for mutable in self.mutable_channels.values():
            mutable: MutableImportance
            imp.append(mutable.importance)
        imp: torch.Tensor = torch.cat(imp, dim=0)
        mask = self.current_mask
        imp = imp[mask]
        imp = imp.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        return imp


class ImportanceConv2d(DynamicConv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device,
                         dtype)

    # module function

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        print(self.mutable_in_channels)
        print(self.mutable_out_channels)
        x = input * self.mutable_in_channels.imp

        x = super().forward(x)

        x = x * self.mutable_out_channels.imp
        return x


@MODELS.register_module()
class ImpUnit(L1MutableChannelUnit):

    def __init__(self,
                 num_channels: int,
                 choice_mode='number',
                 divisor=1,
                 min_value=1,
                 min_ratio=0.9) -> None:
        super().__init__(num_channels, choice_mode, divisor, min_value,
                         min_ratio)
        self.mutable_channel = MutableImportance(num_channels)

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: ImportanceConv2d,
                nn.BatchNorm2d: dynamic_ops.DynamicBatchNorm2d,
                nn.Linear: dynamic_ops.DynamicLinear
            })
        self._register_channel_container(model, MutableImpChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

        # for channel in self.input_related:
        #     if isinstance(channel.module, ImportanceConv2d):
        #         channel.module.add_in_imp(self.mutable)
        # for channel in self.output_related:
        #     if isinstance(channel.module, ImportanceConv2d):
        #         channel.module.add_out_imp(self.mutable)

    def _generate_mask(self, num: int):
        idx = self.mutable_channel.importance.topk(num)[1]
        mask = torch.zeros_like(self.mutable_channel.importance)
        mask.scatter_(0, idx, 1)
        return mask

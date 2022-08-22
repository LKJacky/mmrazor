from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from mmrazor.models.architectures.dynamic_op.default_dynamic_ops import \
    ChannelDynamicOP
from mmrazor.models.mutables.mutable_channel.mutable_channel_unit import \
    MutableChannelUnit
from mmrazor.models.mutables.mutable_channel.mutable_mask import MutableMask
from mmrazor.registry import MODELS


@MODELS.register_module()
class MutableImportance(MutableMask):

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.importance = nn.parameter.Parameter(torch.ones([num_channels]))
        self.importance.requires_grad_()


class ImportanceConv2d(nn.Conv2d, ChannelDynamicOP):

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
        self._mutable_ins = nn.ModuleList()
        self._mutable_outs = nn.ModuleList()

    # importance manage

    def add_in_imp(self, mutable):
        self._mutable_ins.append(mutable)

    def add_out_imp(self, mutable):
        self._mutable_outs.append(mutable)

    def mutable_in_imp(self):
        if len(self._mutable_ins) == 0:
            return torch.ones([self.in_channels])
        else:
            ins = [mutable.importance for mutable in self._mutable_ins]
            ins = torch.cat(ins)
            return ins

    def mutable_out_imp(self):
        if len(self._mutable_outs) == 0:
            return torch.ones([self.out_channels])
        else:
            outs = [mutable.importance for mutable in self._mutable_outs]
            outs = torch.cat(outs)
            return outs

    # module function

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        imp_in = self.mutable_in_imp().reshape([1, -1, 1, 1])
        imp_out = self.mutable_out_imp().reshape([1, -1, 1, 1])
        x = input * imp_in
        x = nn.Conv2d.forward(self, x)
        x = x * imp_out
        return x

    @classmethod
    def copy_from(cls, conv: nn.Conv2d):
        return cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                   conv.stride, conv.padding, conv.dilation, conv.groups)

    def __repr__(self):
        s = super().__repr__()
        return s

    # abstract methods

    def to_static_op(self) -> nn.Module:
        pass

    @property
    def mutable_in(self):
        return super().mutable_in

    @property
    def mutable_out(self):
        return super().mutable_out


@MODELS.register_module()
class ImpUnit(MutableChannelUnit):

    def __init__(self,
                 num_channels,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(num_channels, alias, init_cfg)
        self.mutable = MutableImportance(num_channels)

    def prepare_for_pruning(self):
        for channel in self.input_related:
            if isinstance(channel.module, ImportanceConv2d):
                channel.module.add_in_imp(self.mutable)
        for channel in self.output_related:
            if isinstance(channel.module, ImportanceConv2d):
                channel.module.add_out_imp(self.mutable)

    @classmethod
    def replace_with_dynamic_ops(cls, models: nn.Module):
        return super().replace_with_dynamic_ops(models, [ImportanceConv2d])

    def _generate_mask(self):
        idx = self.mutable.importance.topk(self.num_channels)[1]
        mask = torch.zeros_like(self.mutable.importance)
        mask.scatter_(0, idx, 1)
        return mask

    def apply_mask(self):
        pass

    def set_choice(self, choice: Union[float, int]):
        choice = self._get_int_choice(choice)

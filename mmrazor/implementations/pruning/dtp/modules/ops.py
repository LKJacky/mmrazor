# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor

import mmrazor.models.architectures.dynamic_ops as dynamic_ops
from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from ...chip.collect.ops import CollectMixin
from .mutable_channels import ImpMutableChannelContainer


def ste_forward(x, mask):
    return x.detach() * mask - x.detach() + x


def soft_ceil(x):
    with torch.no_grad():
        x_ceil = torch.ceil(x.detach().data)
    return x_ceil.detach() - x.detach() + x


class QuickFlopMixin:

    def _quick_flop_init(self) -> None:
        self.quick_flop_handlers: list = []
        self.quick_flop_recorded_out_shape: List = []
        self.quick_flop_recorded_in_shape: List = []

    def quick_flop_forward_hook_wrapper(self):
        """Wrap the hook used in forward."""

        def forward_hook(module: QuickFlopMixin, input, output):
            module.quick_flop_recorded_out_shape.append(output.shape)
            module.quick_flop_recorded_in_shape.append(input[0].shape)

        return forward_hook

    def quick_flop_start_record(self: torch.nn.Module) -> None:
        """Start recording information during forward and backward."""
        self.quick_flop_end_record()  # ensure to run start_record only once
        self.quick_flop_handlers.append(
            self.register_forward_hook(self.quick_flop_forward_hook_wrapper()))

    def quick_flop_end_record(self):
        """Stop recording information during forward and backward."""
        for handle in self.quick_flop_handlers:
            handle.remove()
        self.quick_flop_handlers = []

    def quick_flop_reset_recorded(self):
        """Reset the recorded information."""
        self.quick_flop_recorded_out_shape = []
        self.quick_flop_recorded_in_shape = []

    def soft_flop(self):
        raise NotImplementedError()

    @classmethod
    def get_flop(cls, model: nn.Module):
        flops = 0
        if isinstance(model, QuickFlopMixin):
            return model.soft_flop()
        for child in model.children():
            if isinstance(child, QuickFlopMixin):
                flops = flops + child.soft_flop()
            else:
                flops = flops + cls.get_flop(child)
        return flops


class ImpModuleMixin():

    def _imp_init(self):
        self.ste = False

    @property
    def input_imp(
            self: Union[DynamicChannelMixin,
                        'ImpModuleMixin']) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(  # type: ignore # noqa
            'in_channels')  # type: ignore
        imp = mutable.current_imp
        return imp

    @property
    def input_imp_flop(
            self: Union[DynamicChannelMixin,
                        'ImpModuleMixin']) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(  # type: ignore # noqa
            'in_channels')  # type: ignore
        imp = mutable.current_imp_flop
        return imp

    @property
    def output_imp(self: nn.Module) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(
            'out_channels')
        imp = mutable.current_imp
        return imp

    @property
    def output_imp_flop(self: nn.Module) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(
            'out_channels')
        imp = mutable.current_imp_flop
        return imp

    def imp_forward(self, x: torch.Tensor):

        input_imp = self.input_imp
        if len(x.shape) == 4:
            input_imp = input_imp.reshape([1, -1, 1, 1])
        elif len(x.shape) == 2:
            input_imp = input_imp.reshape([1, -1])
        else:
            raise NotImplementedError()
        if self.ste:
            x = ste_forward(x, input_imp)
        else:
            x = x * input_imp
        return x


@torch.jit.script
def soft_mask_sum(mask: torch.Tensor):
    soft = mask.sum()
    hard = (mask >= 0.5).float().sum()
    return hard.detach() - soft.detach() + soft


@torch.jit.script
def conv_soft_flop(input_imp_flop, output_imp_flop, h, w, k1, k2, groups):
    in_c = soft_mask_sum(input_imp_flop)
    out_c = soft_mask_sum(output_imp_flop)
    conv_per_pos = k1 * k2 * in_c * out_c / groups
    flop = conv_per_pos * h * w
    bias_flop = out_c * h * w
    return flop + bias_flop


class ImpConv2d(dynamic_ops.DynamicConv2d, ImpModuleMixin, QuickFlopMixin,
                CollectMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quick_flop_init()
        self._imp_init()
        self._collect_init()

    def forward(self, x: Tensor) -> Tensor:
        x = self.imp_forward(x)
        return nn.Conv2d.forward(self, x)

    def soft_flop(self):
        return conv_soft_flop(self.input_imp_flop, self.output_imp_flop,
                              *self.quick_flop_recorded_out_shape[0][2:],
                              self.kernel_size[0], self.kernel_size[1],
                              self.groups)


class ImpLinear(dynamic_ops.DynamicLinear, ImpModuleMixin, QuickFlopMixin,
                CollectMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quick_flop_init()
        self._imp_init()
        self._collect_init()

    def forward(self, x: Tensor) -> Tensor:
        x = self.imp_forward(x)
        return nn.Linear.forward(self, x)

    def soft_flop(self):
        in_c = soft_mask_sum(self.input_imp_flop)
        out_c = soft_mask_sum(self.output_imp_flop)
        return in_c * out_c


@torch.jit.script
def bn_soft_flop(input_imp_flop, h, w):
    in_c = soft_mask_sum(input_imp_flop)
    return h * w * in_c


class ImpBatchnorm2d(dynamic_ops.DynamicBatchNorm2d, QuickFlopMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quick_flop_init()

    def forward(self, input: Tensor) -> Tensor:
        return nn.BatchNorm2d.forward(self, input)

    def soft_flop(self):
        return bn_soft_flop(self.input_imp_flop,
                            *self.quick_flop_recorded_out_shape[0][2:])

    @property
    def output_imp_flop(self: nn.Module) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(
            'out_channels')
        imp = mutable.current_imp_flop
        return imp

    @property
    def input_imp_flop(
            self: Union[DynamicChannelMixin,
                        'ImpModuleMixin']) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(  # type: ignore # noqa
            'in_channels')  # type: ignore
        imp = mutable.current_imp_flop
        return imp

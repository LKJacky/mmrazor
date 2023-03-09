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


def soft_mask_sum(mask: torch.Tensor):
    soft = mask.sum()
    hard = (mask >= 0.5).float().sum()
    return hard.detach() - soft.detach() + soft


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
        in_c = soft_mask_sum(self.input_imp_flop)
        out_c = soft_mask_sum(self.output_imp_flop)
        conv_per_pos = self.kernel_size[0] * self.kernel_size[
            1] * in_c * out_c / self.groups
        h, w = self.quick_flop_recorded_out_shape[0][2:]
        flop = conv_per_pos * h * w
        bias_flop = out_c * w * w
        return flop + bias_flop


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


class ImpBatchnorm2d(dynamic_ops.DynamicBatchNorm2d, QuickFlopMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quick_flop_init()

    def forward(self, input: Tensor) -> Tensor:
        return nn.BatchNorm2d.forward(self, input)

    def soft_flop(self):
        in_c = soft_mask_sum(self.input_imp_flop)
        h, w = self.quick_flop_recorded_out_shape[0][2:]
        return h * w * in_c

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

# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor

import mmrazor.models.architectures.dynamic_ops as dynamic_ops
from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from .mutable_channels import ImpMutableChannelContainer


def soft_ceil(x):
    with torch.no_grad():
        x_ceil = torch.ceil(x.detach().data)
    return x_ceil.detach() - x.detach() + x


class QuickFlopMixin:

    def _quick_flop_init(self) -> None:
        self.handlers: list = []
        self.recorded_out_shape: List = []
        self.recorded_in_shape: List = []

    def forward_hook_wrapper(self):
        """Wrap the hook used in forward."""

        def forward_hook(module: QuickFlopMixin, input, output):
            module.recorded_out_shape.append(output.shape)
            module.recorded_in_shape.append(input[0].shape)

        return forward_hook

    def start_record(self: torch.nn.Module) -> None:
        """Start recording information during forward and backward."""
        self.end_record()  # ensure to run start_record only once
        self.handlers.append(
            self.register_forward_hook(self.forward_hook_wrapper()))

    def end_record(self):
        """Stop recording information during forward and backward."""
        for handle in self.handlers:
            handle.remove()
        self.handlers = []

    def reset_recorded(self):
        """Reset the recorded information."""
        self.recorded_out_shape = []
        self.recorded_in_shape = []


class ImpModuleMixin():

    @property
    def input_imp(
            self: Union[DynamicChannelMixin,
                        'ImpModuleMixin']) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(  # type: ignore # noqa
            'in_channels')  # type: ignore
        imp = mutable.current_imp
        return imp

    @property
    def output_imp(self: nn.Module) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(
            'out_channels')
        imp = mutable.current_imp
        return imp

    def imp_forward(self, x: torch.Tensor):
        input_imp = self.input_imp
        if len(x.shape) == 4:
            input_imp = input_imp.reshape([1, -1, 1, 1])
        elif len(x.shape) == 2:
            input_imp = input_imp.reshape([1, -1])
        else:
            raise NotImplementedError()
        x = x * input_imp
        return x

    def soft_flop(self):
        # input_imp = self.input_imp
        # output_imp = self.output_imp
        # flop = input_imp.sum() * output_imp.sum()
        return 0.0


class ImpConv2d(dynamic_ops.DynamicConv2d, ImpModuleMixin, QuickFlopMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quick_flop_init()

    def forward(self, x: Tensor) -> Tensor:
        x = self.imp_forward(x)
        return nn.Conv2d.forward(self, x)

    def soft_flop(self):
        in_c = soft_ceil(self.input_imp.sum())
        out_c = soft_ceil(self.output_imp.sum())
        conv_per_pos = self.kernel_size[0] * self.kernel_size[
            1] * in_c * out_c / self.groups
        h, w = self.recorded_out_shape[0][2:]
        flop = conv_per_pos * h * w
        bias_flop = out_c * w * w
        return flop + bias_flop


class ImpLinear(dynamic_ops.DynamicLinear, ImpModuleMixin, QuickFlopMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quick_flop_init()

    def forward(self, x: Tensor) -> Tensor:
        x = self.imp_forward(x)
        return nn.Linear.forward(self, x)

    def soft_flop(self):
        in_c = soft_ceil(self.input_imp.sum())
        out_c = soft_ceil(self.output_imp.sum())
        return in_c * out_c


class ImpBatchnorm2d(dynamic_ops.DynamicBatchNorm2d):

    def forward(self, input: Tensor) -> Tensor:
        return nn.BatchNorm2d.forward(self, input)

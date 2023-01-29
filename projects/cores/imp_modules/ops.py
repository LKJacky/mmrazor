# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from torch import Tensor

import mmrazor.models.architectures.dynamic_ops as dynamic_ops
from mmrazor.models.mutables import (MutableChannelContainer,
                                     SimpleMutableChannel)


class SimpleMutableChannelImp(SimpleMutableChannel):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.imp: torch.Tensor = nn.parameter.Parameter(
            torch.ones([num_channels]))

    @property
    def current_imp(self):
        return self.imp


class ImpMutableChannelContainer(MutableChannelContainer):

    @property
    def current_imp(self):
        """Get current choices."""
        if len(self.mutable_channels) == 0:
            return torch.ones([self.num_channels]).to(self.current_mask.device)
        else:
            self._fill_unregistered_range()
            self._assert_mutables_valid()
            mutable_channels = list(self.mutable_channels.values())
            imps = [mutable.current_imp for mutable in mutable_channels]
            imp = torch.cat(imps)
            return imp


class ImpModuleMixin:

    def input_imp(self: nn.Module) -> torch.Tensor:
        mutable: ImpMutableChannelContainer = self.get_mutable_attr(
            'in_channels')
        imp = mutable.current_imp
        imp = imp.unsqueeze(0)
        for _ in range(len(self.weight.shape) - 2):
            imp = imp.unsqueeze(-1)
        return imp

    def imp_forward(self, x):
        imp = self.input_imp().to(x.device)
        x = x * imp
        return x


class ImpConv2d(dynamic_ops.DynamicConv2d, ImpModuleMixin):

    def forward(self, x: Tensor) -> Tensor:
        x = self.imp_forward(x)
        return nn.Conv2d.forward(self, x)


class ImpLinear(dynamic_ops.DynamicLinear, ImpModuleMixin):

    def forward(self, x: Tensor) -> Tensor:
        x = self.imp_forward(x)
        return nn.Linear.forward(self, x)


class ImpBatchnorm2d(dynamic_ops.DynamicBatchNorm2d):

    def forward(self, input: Tensor) -> Tensor:
        return nn.BatchNorm2d.forward(self, input)

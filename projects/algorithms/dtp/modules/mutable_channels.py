# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from mmrazor.models.mutables import (MutableChannelContainer,
                                     SimpleMutableChannel)


def soft_clip(x: torch.Tensor, min, max):
    with torch.no_grad():
        min = x.new_tensor([min])
        max = x.new_tensor([max])
        y = torch.clip(x, min, max)
    return y.detach() - x.detach() + x


def dtopk(x: torch.Tensor, e: torch.Tensor, lamda=1.0):
    # add min or max
    e = soft_clip(e, 1 / x.numel(), 1)

    y: torch.Tensor = -(x - e) * x.numel() * lamda
    s = y.sigmoid()
    return s


def grad_clip_wrapper(abs):

    def grad_clip_hook(grad: torch.Tensor):
        if abs < 0:
            return None
        else:
            return torch.clamp(grad, -abs, abs)

    return grad_clip_hook


# mutable channels


class DTPMutableChannelImp(SimpleMutableChannel):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)

        index = torch.linspace(0, 1, num_channels)
        self.register_buffer('index', index)
        self.index: torch.Tensor
        self.lamda = 1.0

    @property
    def current_imp(self):
        w = dtopk(self.index, self.e, self.lamda)
        with torch.no_grad():
            self.mask.data = (w >= 0.5).float()
        return w

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.num_channels, 1.0)


class PASMutableChannel(SimpleMutableChannel):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.imp = nn.parameter.Parameter(
            torch.tensor([0.6] * self.num_channels), requires_grad=False)

    @property
    def current_imp(self):
        with torch.no_grad():
            mask = self.imp > 0.5
            self.mask.data = mask.float()
        imp = mask.float().detach() - self.imp.detach() + self.imp

        return imp

    def limit_value(self):
        pass


class ImpMutableChannelContainer(MutableChannelContainer):

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.register_buffer(
            '_tmp_imp', torch.ones([self.num_channels]), persistent=False)
        self._tmp_imp: torch.Tensor

    @property
    def current_imp(self):
        """Get current choices."""
        if len(self.mutable_channels) == 0:
            return self._tmp_imp
        else:
            self._fill_unregistered_range()
            self._assert_mutables_valid()
            mutable_channels = list(self.mutable_channels.values())
            imps = [mutable.current_imp for mutable in mutable_channels]
            imp = torch.cat(imps)
            return imp

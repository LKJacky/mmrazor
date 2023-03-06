# Copyright (c) OpenMMLab. All rights reserved.

import math

import torch
import torch.nn as nn

from mmrazor.models.mutables import (MutableChannelContainer,
                                     SimpleMutableChannel)
from mmrazor.utils import RuntimeInfo


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


def grad_adjust_wrapper(mode=None):

    def current_grad_lr():
        ratio = RuntimeInfo().epoch() / RuntimeInfo().max_epochs()
        return (math.cos(ratio * math.pi) + 1) / 2

    def grad_adjust_hook(grad: torch.Tensor):
        if mode is None:
            return None
        elif mode == 'cos':
            return current_grad_lr() * grad
        else:
            raise NotImplementedError()

    return grad_adjust_hook


# mutable channels


class BaseDTPMutableChannel(SimpleMutableChannel):

    @property
    def current_imp(self):
        raise NotImplementedError()

    @property
    def current_imp_flop(self):
        raise NotImplementedError()

    @torch.no_grad()
    def limit_value(self):
        raise NotImplementedError()


class DTPMutableChannelImp(SimpleMutableChannel):

    def __init__(self, num_channels: int, delta_limit=-1, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)

        index = torch.linspace(0, 1, num_channels)
        self.register_buffer('index', index)
        self.index: torch.Tensor
        self.lamda = 1.0

        self.delta_limit = delta_limit

    @property
    def current_imp(self):
        w = dtopk(self.index, self.e, self.lamda)
        with torch.no_grad():
            self.mask.data = (w >= 0.5).float()
        return w

    @property
    def current_imp_flop(self):
        return self.current_imp

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.num_channels, 1.0)
        if self.delta_limit > 0:
            self.e.data = torch.clamp(self.e, self.pre_e - self.delta_limit,
                                      self.pre_e + self.delta_limit)  # noqa

    @torch.no_grad()
    def save_info(self):
        self.pre_e = self.e.detach().clone()


class DTPAdaptiveMutableChannelImp(SimpleMutableChannel):

    def __init__(self, num_channels: int, delta_limit=-1, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.v = nn.parameter.Parameter(
            torch.tensor([0.0] * self.num_channels), requires_grad=False)
        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)

        self.index: torch.Tensor
        self.lamda = self.num_channels
        self.delta_limit = delta_limit

    def get_importance(self, v: torch.Tensor, e: torch.Tensor):
        vm = v.unsqueeze(-1) - v.unsqueeze(0)
        vm = (vm >= 0).float() - vm.detach() + vm
        v = vm.mean(dim=-1)
        return dtopk(v, e)

    @property
    def current_imp(self):
        if self.e.requires_grad is False and self.v.requires_grad is False:
            return self.mask
        else:
            dis = self.v.max() - self.v.min()
            if dis.abs() == 0.0:
                return self.v.new_ones(
                    self.v.shape) - self.v.detach() + self.v + 0 * self.e
            else:
                imp = self.get_importance(self.v, self.e)
                with torch.no_grad():
                    self.mask.data = (imp >= 0.5).float()
                return imp

    @property
    def current_imp_flop(self):
        dis = self.v.max() - self.v.min()
        if dis == 0:
            return self.v.new_ones(
                self.v.shape) - self.v.detach() + self.v + 0 * self.e
        else:
            imp = self.get_importance(self.v.detach(), self.e)
            return imp

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.num_channels, 1.0)
        # if self.v.max() != self.v.min():
        #     self.v.data = (self.v - self.v.min()) / (
        #         self.v.max() - self.v.min())
        # self.v.data = torch.clamp(self.v, 0.0, 1.0)
        # if self.delta_limit > 0:
        #     self.e.data = torch.clamp(self.e, self.pre_e - self.delta_limit,
        #                               self.pre_e + self.delta_limit)  # noqa
        pass

    @torch.no_grad()
    def save_info(self):
        self.pre_e = self.e.detach().clone()


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

    @property
    def current_imp_flop(self):
        return self.current_imp


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

    @property
    def current_imp_flop(self):
        """Get current choices."""
        if len(self.mutable_channels) == 0:
            return self._tmp_imp
        else:
            self._fill_unregistered_range()
            self._assert_mutables_valid()
            mutable_channels = list(self.mutable_channels.values())
            imps = [mutable.current_imp_flop for mutable in mutable_channels]
            imp = torch.cat(imps)
            return imp

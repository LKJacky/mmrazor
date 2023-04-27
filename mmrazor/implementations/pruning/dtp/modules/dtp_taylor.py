# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmengine.dist import all_reduce

from mmrazor.registry import MODELS
from .mutable_channels import BaseDTPMutableChannel
from .unit import BaseDTPUnit

# dtp with taylor importance base dtp with adaptive importance


@torch.jit.script
def dtopk(x: torch.Tensor, e: torch.Tensor, lamda: float = 1.0):
    # add min or max
    # e = soft_clip(e, 1 / x.numel(), 1.0)

    y: torch.Tensor = -(x - e) * x.numel() * lamda
    s = y.sigmoid()
    return s


@torch.jit.script
def dtp_get_importance(v: torch.Tensor,
                       e: torch.Tensor,
                       lamda: float = 1.0,
                       space_min: float = 0,
                       space_max: float = 1.0):
    vm = v.unsqueeze(-1) - v.unsqueeze(-2)
    vm = (vm >= 0).float() - vm.detach() + vm
    v_union = vm.mean(dim=-1)  # big to small
    v_union = 1 - v_union
    if space_max != 1.0 or space_min != 0:
        v_union = v_union * (space_max - space_min) + space_min
    imp = dtopk(v_union, e, lamda=lamda)  # [0,1]
    return imp


def taylor_backward_hook_wrapper(module: 'DTPTMutableChannelImp', input):

    def taylor_backward_hook(grad):
        with torch.no_grad():
            module.update_taylor(input, grad)

    return taylor_backward_hook


class DMSMutableMixIn():

    def _dms_mutable_mixin_init(self, num_elem):
        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)

        taylor = torch.zeros([num_elem])
        self.register_buffer('taylor', taylor)
        self.taylor: torch.Tensor

        self.decay = 0.99
        self.requires_grad_(False)

    @property
    def current_imp(self):
        e_imp = dtp_get_importance(self.taylor, self.e)
        if self.training and e_imp.requires_grad:
            e_imp.register_hook(
                taylor_backward_hook_wrapper(self, e_imp.detach()))
        if self.training:
            with torch.no_grad():
                self.mask.data = (e_imp >= 0.5).float()
        return e_imp

    @property
    def current_imp_flop(self):
        e_imp = dtp_get_importance(self.taylor, self.e)
        return e_imp

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.num_channels, 1.0)

    @torch.no_grad()
    def update_taylor(self, input, grad):
        new_taylor = (input * grad)**2
        all_reduce(new_taylor)
        self.taylor = self.taylor * self.decay + (1 - self.decay) * new_taylor

    def dump_chosen(self):
        pass

    def fix_chosen(self, chosen=None):
        pass

    @property
    def activated_channels(self):
        return self.mask.bool().sum().item()

    def info(self):
        return (f'taylor: {self.taylor.min().item():.3f}\t'
                f'{self.taylor.max().item():.3f}\t'
                f'e: {self.e.item():.3f}')  # noqa


class DTPTMutableChannelImp(BaseDTPMutableChannel):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)

        taylor = torch.zeros([num_channels])
        self.register_buffer('taylor', taylor)
        self.taylor: torch.Tensor

        self.decay = 0.99
        self.requires_grad_(False)

    @property
    def current_imp(self):
        e_imp = dtp_get_importance(self.taylor, self.e)
        if self.training and e_imp.requires_grad:
            e_imp.register_hook(
                taylor_backward_hook_wrapper(self, e_imp.detach()))
        if self.training:
            with torch.no_grad():
                self.mask.data = (e_imp >= 0.5).float()
        return e_imp

    @property
    def current_imp_flop(self):
        e_imp = dtp_get_importance(self.taylor, self.e)
        return e_imp

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.num_channels, 1.0)

    @torch.no_grad()
    def update_taylor(self, input, grad):
        new_taylor = (input * grad)**2
        all_reduce(new_taylor)
        self.taylor = self.taylor * self.decay + (1 - self.decay) * new_taylor


@MODELS.register_module()
class DTPTUnit(BaseDTPUnit):

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__(num_channels)
        self.mutable_channel: DTPTMutableChannelImp = DTPTMutableChannelImp(
            self.num_channels)
        self.requires_grad_(False)

    @torch.no_grad()
    def info(self) -> str:
        return (f'taylor: {self.mutable_channel.taylor.min().item():.3f}\t'
                f'{self.mutable_channel.taylor.max().item():.3f}\t'
                f'e: {self.mutable_channel.e.item():.3f}')  # noqa

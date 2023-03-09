# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmengine.dist import all_reduce

from mmrazor.registry import MODELS
from .mutable_channels import BaseDTPMutableChannel, dtp_get_importance
from .unit import BaseDTPUnit

# dtp with taylor importance base dtp with adaptive importance


def taylor_backward_hook_wrapper(module: 'DTPTMutableChannelImp'):

    def taylor_backward_hook(grad):
        module.update_taylor(grad)

    return taylor_backward_hook


class DTPTMutableChannelImp(BaseDTPMutableChannel):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)

        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)

        self.imp = nn.parameter.Parameter(torch.ones([num_channels]))

        taylor = torch.zeros([num_channels])
        self.register_buffer('taylor', taylor)
        self.taylor: torch.Tensor

        self.imp.register_hook(taylor_backward_hook_wrapper(self))
        self.decay = 0.99
        self.requires_grad_(False)

    @property
    def current_imp(self):
        e_imp = dtp_get_importance(self.taylor, self.e)
        w = e_imp * self.imp
        if self.training:
            with torch.no_grad():
                self.mask.data = (e_imp >= 0.5).float()
        return w

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.num_channels, 1.0)

    @torch.no_grad()
    def update_taylor(self, grad):
        new_taylor = (self.imp * grad)**2
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
        return (f'imp gate: {self.mutable_channel.imp.min().item():.3f}\t'
                f'{self.mutable_channel.imp.max().item():.3f}\t'
                f'taylor: {self.mutable_channel.taylor.min().item():.3f}\t'
                f'{self.mutable_channel.taylor.max().item():.3f}\t'
                f'e: {self.mutable_channel.e.item():.3f}')  # noqa

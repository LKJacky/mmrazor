# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmrazor.registry import MODELS
from .dtp_taylor import DTPTMutableChannelImp, taylor_backward_hook_wrapper
from .unit import BaseDTPUnit

# dtp with taylor importance base dtp with adaptive importance


@torch.jit.script
def get_importance(v: torch.Tensor, e: torch.Tensor):
    e_s = e.sigmoid()
    vm = v.unsqueeze(-1) - v.unsqueeze(0)
    vm = (vm >= 0).float() - vm.detach() + vm
    v_union = vm.mean(dim=-1)  # big to small
    v_union = 1 - v_union
    mask = (v_union < e_s).float()
    return mask.detach() - e.detach() + e


class LEAPMutableChannel(DTPTMutableChannelImp):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.e.fill_(10)

    @property
    def current_imp(self):
        e_imp = get_importance(self.taylor, self.e)
        if self.training and e_imp.requires_grad:
            e_imp.register_hook(taylor_backward_hook_wrapper(self, e_imp))
        if self.training:
            with torch.no_grad():
                self.mask.data = (e_imp >= 0.5).float()
        return e_imp

    @property
    def current_imp_flop(self):
        e_imp = get_importance(self.taylor, self.e)
        return e_imp

    def limit_value(self):
        pass


@MODELS.register_module()
class LEAPUnit(BaseDTPUnit):

    def __init__(
        self,
        num_channels: int,
    ) -> None:
        super().__init__(num_channels)
        self.mutable_channel: LEAPMutableChannel = LEAPMutableChannel(
            self.num_channels)
        self.requires_grad_(False)

    def info(self) -> str:
        return (f'taylor: {self.mutable_channel.taylor.min().item():.3f}\t'
                f'{self.mutable_channel.taylor.max().item():.3f}\t'
                f'e_s: {self.mutable_channel.e.sigmoid().item():.3f}\t'
                f'e: {self.mutable_channel.e.item():.3f}')  # noqa

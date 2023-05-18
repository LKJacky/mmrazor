# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmengine.dist import all_reduce

from mmrazor.models.mutables import DerivedMutable
from mmrazor.registry import MODELS
from .mutable_channels import SimpleMutableChannel
from .unit import BaseDTPUnit

MaskThreshold = 0.5
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


class DrivedDTPMutableChannelImp(DerivedMutable):

    def __init__(self,
                 choice_fn,
                 mask_fn,
                 expand_ratio,
                 source_mutables=None,
                 alias=None,
                 init_cfg=None) -> None:
        super().__init__(choice_fn, mask_fn, source_mutables, alias, init_cfg)
        self.expand_ratio = expand_ratio

    @property
    def current_imp(self):
        mutable = list(self.source_mutables)[0]
        mask = mutable.current_imp
        mask = torch.unsqueeze(
            mask,
            -1).expand(list(mask.shape) + [self.expand_ratio]).flatten(-2)
        return mask

    @property
    def current_imp_flop(self):
        mutable = list(self.source_mutables)[0]
        mask = mutable.current_imp_flop
        mask = torch.unsqueeze(
            mask,
            -1).expand(list(mask.shape) + [self.expand_ratio]).flatten(-2)
        return mask


class DMSMutableMixIn():

    def _dms_mutable_mixin_init(self, num_elem):

        self.use_tayler = True

        self.e = nn.parameter.Parameter(
            torch.tensor([1.0]), requires_grad=False)

        taylor = torch.zeros([num_elem])
        self.register_buffer('taylor', taylor)
        self.taylor: torch.Tensor

        self.decay = 0.99
        self.lda = 1.0
        self.requires_grad_(False)

    @property
    def current_imp(self):
        if self.taylor.max() == self.taylor.min():
            e_imp = torch.ones_like(self.taylor, requires_grad=True)
        else:
            e_imp = dtp_get_importance(self.taylor, self.e, lamda=self.lda)
        if self.training and e_imp.requires_grad and self.use_tayler:
            e_imp.register_hook(
                taylor_backward_hook_wrapper(self, e_imp.detach()))
        if self.training:
            with torch.no_grad():
                self.mask.data = (e_imp >= MaskThreshold).float()
        return e_imp

    @property
    def current_imp_flop(self):
        e_imp = dtp_get_importance(self.taylor, self.e)
        return e_imp

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 0, 1.0)

    @torch.no_grad()
    def update_taylor(self, input, grad):
        new_taylor = (input * grad)**2
        all_reduce(new_taylor)
        if new_taylor.max() != new_taylor.min():
            self.taylor = self.taylor * self.decay + (1 -
                                                      self.decay) * new_taylor

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

    def expand_mutable_channel(self, expand_ratio):

        def _expand_mask():
            mask = self.current_mask
            mask = torch.unsqueeze(
                mask, -1).expand(list(mask.shape) + [expand_ratio]).flatten(-2)
            return mask

        return DrivedDTPMutableChannelImp(_expand_mask, _expand_mask,
                                          expand_ratio, [self])

    @torch.no_grad()
    def to_index_importance(self):
        self.use_tayler = False
        self.taylor.data = 1 - torch.linspace(
            0, 1, self.taylor.numel(), device=self.taylor.device)


class DTPTMutableChannelImp(SimpleMutableChannel, DMSMutableMixIn):

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self._dms_mutable_mixin_init(self.num_channels)

    def fix_chosen(self, chosen=None):
        return super().fix_chosen(chosen)


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

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.dist import all_reduce

from mmrazor.models.mutables import BaseMutable, SimpleMutableChannel
from ...dtp.modules.dtp_taylor import (DMSMutableMixIn, dtp_get_importance,
                                       taylor_backward_hook_wrapper)

BlockThreshold = 0.5


class MutableBlocks(BaseMutable):

    def __init__(self, num_blocks) -> None:
        super().__init__()

        self.num_blocks = num_blocks

        mask = torch.ones([num_blocks])
        self.register_buffer('mask', mask)
        self.mask: torch.Tensor

        self.e = nn.parameter.Parameter(torch.tensor(1.0), requires_grad=False)

        taylor = torch.zeros([num_blocks])
        self.register_buffer('taylor', taylor)
        self.taylor: torch.Tensor

        self.decay = 0.99
        self.requires_grad_(False)

        self.flop_scale_converter = None

    def block_scale_fun_wrapper(self, i):

        def scale():
            scale = self.current_imp[i]
            if self.flop_scale_converter is None:
                return scale
            else:
                return self.flop_scale_converter(scale)

        return scale

    def block_flop_scale_fun_wrapper(self, i):

        def scale():
            return self.current_imp[i]

        return scale

    @property
    def current_imp(self):
        imp = dtp_get_importance(self.taylor, self.e, lamda=4.0)

        if self.training and imp.requires_grad:
            imp.register_hook(taylor_backward_hook_wrapper(self, imp.detach()))
            with torch.no_grad():
                self.mask.data = (imp >= BlockThreshold).float()
        return imp

    @property
    def current_imp_flop(self):
        raise NotImplementedError()

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clip(self.e.data, 1 / self.num_blocks, 1.0)

    @torch.no_grad()
    def update_taylor(self, input, grad):
        new_taylor = (input * grad)**2
        all_reduce(new_taylor)
        self.taylor = self.taylor * self.decay + (1 - self.decay) * new_taylor

    def info(self):

        def get_mask_str():
            mask_str = ''
            for i in range(self.num_blocks):
                if self.mask[i] == 1:
                    mask_str += '1'
                else:
                    mask_str += '0'
            return mask_str

        return (
            f'mutable_block: {self.num_blocks} \t {self.e.item():.3f}, \t'
            f'self.taylor: \t{self.taylor.min().item():.3f}\t{self.taylor.max().item():.3f}\t'  # noqa
            f'mask:\t{get_mask_str()}\t')

    # inherit from BaseMutable

    @property
    def current_choice(self):
        return super().current_choice

    def fix_chosen(self, chosen) -> None:
        return super().fix_chosen(chosen)

    def dump_chosen(self):
        return super().dump_chosen()


class MutableHead(BaseMutable, DMSMutableMixIn):

    def __init__(self, num_heads) -> None:
        super().__init__()
        self.num_heads = num_heads
        self._dms_mutable_mixin_init(num_heads)

        mask = torch.ones([num_heads])
        self.register_buffer('mask', mask)
        self.mask: torch.Tensor

        self.flop_scale_converter = None

    @property
    def current_imp(self):
        if self.flop_scale_converter is None:
            return super().current_imp
        else:
            return self.flop_scale_converter(super().current_imp)

    @property
    def current_imp_flop(self):
        if self.flop_scale_converter is None:
            return super().current_imp_flop
        else:
            return self.flop_scale_converter(super().current_imp_flop)

    def dump_chosen(self):
        pass

    def fix_chosen(self, chosen=None):
        pass

    @property
    def current_choice(self):
        return None

    @torch.no_grad()
    def limit_value(self):
        self.e.data = torch.clamp(self.e, 1 / self.num_heads, 1.0)

    def info(self):

        def get_mask_str():
            mask_str = ''
            for i in range(self.num_heads):
                if self.mask[i] == 1:
                    mask_str += '1'
                else:
                    mask_str += '0'
            return mask_str

        return super().info() + f'\t{get_mask_str()}\t'


class MutableChannelForHead(BaseMutable, DMSMutableMixIn):

    def __init__(self, num_channels, num_heads) -> None:
        super().__init__()
        self._dms_mutable_mixin_init(num_channels)
        self.num_head = num_heads
        self.num_channels = num_channels

        self.taylor = self.taylor.reshape([num_heads, -1])

        mask = torch.ones([num_channels])
        self.register_buffer('mask', mask)
        self.mask: torch.Tensor
        self.mask = self.mask.reshape([num_heads, -1])

    def dump_chosen(self):
        pass

    def fix_chosen(self, chosen=None):
        pass

    @property
    def current_choice(self):
        return None


class MutableChannelWithHead(SimpleMutableChannel):

    def __init__(self, mutable_head: MutableHead,
                 mutable_channel: MutableChannelForHead) -> None:
        super().__init__(mutable_channel.num_channels)

        self.mutable_head = mutable_head
        self.mutable_channel = mutable_channel

    @property
    def current_imp(self):
        channel_imp = self.mutable_channel.current_imp
        head_imp = self.mutable_head.current_imp
        imp = head_imp.unsqueeze(-1) * channel_imp
        imp = imp.flatten()
        return imp

    @property
    def current_imp_flop(self):
        current_imp_flop = self.mutable_channel.current_imp_flop
        head_imp = self.mutable_head.current_imp_flop
        imp = head_imp.unsqueeze(-1) * current_imp_flop
        imp = imp.flatten()
        return imp

    @property
    def current_mask(self):
        channel = self.mutable_channel.mask
        head = self.mutable_head.mask.unsqueeze(-1)

        return (channel * head).bool().flatten()

    @torch.no_grad()
    def limit_value(self):
        self.mutable_head.limit_value()
        self.mutable_channel.limit_value()

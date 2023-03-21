# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.dist import all_reduce

from mmrazor.models.mutables import BaseMutable
from ...dtp.modules.dtp_taylor import (dtp_get_importance,
                                       taylor_backward_hook_wrapper)


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

    def block_scale_fun_wrapper(self, i):

        def scale():
            return self.current_imp[i]

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
                self.mask.data = (imp >= 0.5).float()
        return imp

    @property
    def current_imp_flop(self):
        e_imp = dtp_get_importance(self.taylor, self.e)
        return e_imp

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
            f'mutable_block_{self.num_blocks}:\t{self.e.item():.3f}, \t'
            f'self.taylor: {self.taylor.min().item():.3f}\t{self.taylor.max().item():.3f}\t'  # noqa
            f'{get_mask_str()}')

    # inherit from BaseMutable

    @property
    def current_choice(self):
        return super().current_choice

    def fix_chosen(self, chosen) -> None:
        return super().fix_chosen(chosen)

    def dump_chosen(self):
        return super().dump_chosen()

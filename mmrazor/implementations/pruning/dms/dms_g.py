# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.dist import all_reduce

from mmrazor.models.mutables import BaseMutable
from mmrazor.registry import MODELS
from ..dtp.modules.dtp_taylor import dtp_get_importance
from .core.mutable import MutableBlocks
from .core.mutator import BlockInitialer, DMSMutator
from .core.op import DynamicBlockMixin, DynamicStage

# differential model scale gradually.


def taylor_backward_hook_wrapper(module: 'GraduallyMutableBlocks', input, i):

    def taylor_backward_hook(grad):
        with torch.no_grad():
            module.update_taylor(input, grad, i)

    return taylor_backward_hook


class GraduallyMutableBlocks(MutableBlocks):

    def __init__(self, num_blocks, num_channel) -> None:
        super(BaseMutable, self).__init__()

        self.num_blocks = num_blocks
        self.num_channel = num_channel

        mask = torch.ones([num_blocks])
        self.register_buffer('mask', mask)
        self.mask: torch.Tensor

        self.e = nn.parameter.Parameter(torch.tensor(1.0), requires_grad=False)

        taylor = torch.zeros([num_blocks, num_channel])
        self.register_buffer('taylor', taylor)
        self.taylor: torch.Tensor

        self.decay = 0.99
        self.requires_grad_(False)

    def block_scale_fun_wrapper(self, i):

        def scale():
            return self.get_current_imp(i)

        return scale

    def block_flop_scale_fun_wrapper(self, i):

        def scale():
            imp = self.get_current_imp(i)
            hard_scale = (imp >= 0.5).any().float()
            scale = hard_scale.detach() - imp.mean().detach() + imp.mean()
            return scale

        return scale

    @property
    def current_imp(self):
        raise NotImplementedError()

    def get_current_imp(self, i):
        i = i % self.num_blocks
        space_min = i / self.num_blocks
        space_max = (i + 1) / self.num_blocks
        imp = dtp_get_importance(
            self.taylor[i],
            self.e,
            lamda=self.num_blocks,
            space_min=space_min,
            space_max=space_max)
        if self.training and imp.requires_grad:
            imp.register_hook(
                taylor_backward_hook_wrapper(self, imp.detach(), i))
            with torch.no_grad():
                self.mask.data[i] = (imp >= 0.5).any().float()
        return imp

    @property
    def current_imp_flop(self):
        raise NotImplementedError()

    @torch.no_grad()
    def update_taylor(self, input, grad, i):
        new_taylor = (input * grad)**2
        all_reduce(new_taylor)
        self.taylor[i] = self.taylor[i] * self.decay + (
            1 - self.decay) * new_taylor

    @torch.no_grad()
    def info(self):

        def get_mask_str():
            mask_str = ''
            for i in range(self.num_blocks):
                if self.mask[i] == 1:
                    mask_str += '1'
                else:
                    mask_str += '0'
            return mask_str

        imp = self.get_current_imp(0)
        imp1 = self.get_current_imp(-1)

        return (
            f'mutable_block_{self.num_blocks}:\t{self.e.item():.3f}, \t'
            f'self.taylor:\t{self.taylor.min().item():.3f}\t{self.taylor.max().item():.3f}\t'  # noqa
            f'imp:\t {imp.max().item():.3f}\t{imp.min().item():.3f},\t{imp1.max().item():.3f}\t{imp1.min().item():.3f}\t'  # noqa
            f'{get_mask_str()}')


class GraduallyDynamicStaget(DynamicStage):

    def __init__(self, *args):
        super().__init__(*args)

        for module in self.modules():
            if isinstance(module, DynamicBlockMixin):
                block0 = module

        self.mutable_attrs = {}
        self.register_mutable_attr(
            'mutable_blocks',
            GraduallyMutableBlocks(
                len(list(self.removable_block)),
                num_channel=block0.out_channel))
        self.mutable_blocks: GraduallyMutableBlocks

        for i, block in enumerate(self.removable_block):
            block._scale_func = self.mutable_blocks.block_scale_fun_wrapper(i)


@MODELS.register_module()
class DMSGMutator(DMSMutator):

    def __init__(self, *args, **kwargs):
        DMSMutator.__init__(self, *args, **kwargs)
        self.block_initializer = BlockInitialer(
            dynamic_statge_module=GraduallyDynamicStaget)

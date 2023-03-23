# dms by channel

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dist import all_reduce

from mmrazor.registry import MODELS
from ..dtp.modules.dtp_taylor import dtp_get_importance
from .core.mutable import MutableBlocks
from .core.mutator import BlockInitialer, DMSMutator
from .core.op import DynamicBlockMixin, DynamicStage

# differential model scale gradually.


def taylor_backward_hook_wrapper(module: 'ChannelMutableBlocks', input, i):

    def taylor_backward_hook(grad):
        with torch.no_grad():
            module.update_taylor(input, grad, i)

    return taylor_backward_hook


class ChannelMutableBlocks(MutableBlocks):

    def __init__(self, num_blocks, num_channel) -> None:
        super().__init__(num_blocks)

        self.num_channel = num_channel

        taylor = torch.zeros([num_blocks, num_channel])
        self.register_buffer('taylor', taylor)
        self.taylor: torch.Tensor

    def block_scale_fun_wrapper(self, i):

        def scale():
            return self.get_current_imp(i)

        return scale

    def block_flop_scale_fun_wrapper(self, i):

        def scale():
            imp = self.get_current_imp(i)
            scale = imp.mean()
            if self.flop_scale_converter is None:
                return scale
            else:
                return self.flop_scale_converter(scale)

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

    @torch.no_grad()
    def update_taylor(self, input, grad, i):
        new_taylor = (input * grad)**2
        all_reduce(new_taylor)
        self.taylor[i] = self.taylor[i] * self.decay + (
            1 - self.decay) * new_taylor

    @torch.no_grad()
    def info(self):
        info = super().info()

        imp = self.get_current_imp(0)
        imp1 = self.get_current_imp(-1)

        return info + (
            f'channel_imp:\t {imp.max().item():.3f}\t{imp.min().item():.3f}\t'
            f'{imp1.max().item():.3f}\t{imp1.min().item():.3f}\t')  # noqa


class ChannelDynamicStaget(DynamicStage):

    def __init__(self, *args):
        super().__init__(*args)

        for module in self.modules():
            if isinstance(module, DynamicBlockMixin):
                block0 = module

        self.register_mutable_attr(
            'mutable_blocks',
            ChannelMutableBlocks(
                len(list(self.removable_block)),
                num_channel=block0.out_channel))
        self.mutable_blocks: ChannelMutableBlocks

        self.prepare_blocks()


@MODELS.register_module()
class DMSCMutator(DMSMutator):

    def __init__(self, *args, **kwargs):
        DMSMutator.__init__(self, *args, **kwargs)
        self.block_initializer = BlockInitialer(
            dynamic_statge_module=ChannelDynamicStaget)

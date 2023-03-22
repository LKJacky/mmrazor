# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import MODELS, TASK_UTILS
from ..dtp.modules.dtp_adaptive import DTPAScheduler
from .core.mutator import BlockInitialer, DMSMutator
from .core.op import DynamicBlockMixin, DynamicStage
from .dms_g import GraduallyMutableBlocks

# differential model scale gradually softly.


class GraduallySoftMutableBlocks(GraduallyMutableBlocks):

    def __init__(self, num_blocks, num_channel) -> None:
        super().__init__(num_blocks, num_channel)
        self.soft_flop_mode = True

    def block_flop_scale_fun_wrapper(self, i):

        def scale():
            if self.soft_flop_mode:
                imp: torch.Tensor = self.get_current_imp(i)
                return imp.mean()
            else:
                imp = self.get_current_imp(i)
                hard_scale = (imp >= 0.5).any().float()
                scale = hard_scale.detach() - imp.mean().detach() + imp.mean()
                return scale

        return scale


class GraduallySoftDynamicStaget(DynamicStage):

    def __init__(self, *args):
        super().__init__(*args)

        for module in self.modules():
            if isinstance(module, DynamicBlockMixin):
                block0 = module

        self.mutable_attrs = {}
        self.register_mutable_attr(
            'mutable_blocks',
            GraduallySoftMutableBlocks(
                len(list(self.removable_block)),
                num_channel=block0.out_channel))
        self.mutable_blocks: GraduallySoftMutableBlocks

        for i, block in enumerate(self.removable_block):
            block._scale_func = self.mutable_blocks.block_scale_fun_wrapper(i)
            block._flop_scale_func = self.mutable_blocks.block_flop_scale_fun_wrapper(  # noqa
                i)


@MODELS.register_module()
class DMSGSMutator(DMSMutator):

    def __init__(self, *args, **kwargs):
        DMSMutator.__init__(self, *args, **kwargs)
        self.block_initializer = BlockInitialer(
            dynamic_statge_module=GraduallySoftDynamicStaget)

    def soft_flop_mode_to(self, mode=False):
        for module in self.block_mutables:
            module.soft_flop_mode = mode


@TASK_UTILS.register_module()
class DMSGSScheduler(DTPAScheduler):
    mutator: DMSGSMutator

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        self.mutator.limit_value()
        if iter < (self.decay_ratio) * max_iters:
            self.mutator.ratio_train()
            self.mutator.soft_flop_mode_to(True)
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.mutator.ratio_train()
            self.mutator.soft_flop_mode_to(False)
        else:
            self.mutator.requires_grad_(False)


@TASK_UTILS.register_module()
class DMSGS_E_Scheduler(DMSGSScheduler):

    def current_target(self, iter, epoch, max_iters, max_epochs):

        def get_target(ratio):
            assert 0 <= ratio <= 1
            return 1 - (1 - self.flops_target) * ratio

        if iter < self.decay_ratio * max_iters:
            ratio = epoch / (max_epochs * self.decay_ratio)
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            ratio = 1.0
        else:
            ratio = 1.0
        return get_target(ratio)

# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import MODELS, TASK_UTILS
from .core.mutator import BlockInitialer, DMSMutator
from .core.op import DynamicBlockMixin, DynamicStage
from .dms_g import GraduallyMutableBlocks
from .dms_gs import DMSGS_E_Scheduler

# differential model scale gradually softly.


@torch.jit.script
def _scale_polarize(s: torch.Tensor, lam: float):
    s = torch.sigmoid(s * lam) - 0.5
    return s


@torch.jit.script
def scale_polarize(scale, lam=1.0):

    return _scale_polarize(scale, lam) / _scale_polarize(torch.ones([1]),
                                                         lam).item()


class GraduallySoft_D_MutableBlocks(GraduallyMutableBlocks):

    def __init__(self, num_blocks, num_channel) -> None:
        super().__init__(num_blocks, num_channel)
        self.lamda = 0

    def block_flop_scale_fun_wrapper(self, i):

        def scale():
            imp: torch.Tensor = self.get_current_imp(i)
            s = imp.mean()
            s = scale_polarize(s, max(self.lamda * self.num_channel * 2, 1))
            return s

        return scale

    def info(self):
        return super().info() + f'\tlamda: \t{self.lamda}'


class GraduallySoft_D_DynamicStaget(DynamicStage):

    def __init__(self, *args):
        super().__init__(*args)

        for module in self.modules():
            if isinstance(module, DynamicBlockMixin):
                block0 = module

        self.mutable_attrs = {}
        self.register_mutable_attr(
            'mutable_blocks',
            GraduallySoft_D_MutableBlocks(
                len(list(self.removable_block)),
                num_channel=block0.out_channel))
        self.mutable_blocks: GraduallySoft_D_MutableBlocks

        for i, block in enumerate(self.removable_block):
            block._scale_func = self.mutable_blocks.block_scale_fun_wrapper(i)
            block._flop_scale_func = self.mutable_blocks.block_flop_scale_fun_wrapper(  # noqa
                i)


@MODELS.register_module()
class DMSGS_D_Mutator(DMSMutator):

    def __init__(self, *args, **kwargs):
        DMSMutator.__init__(self, *args, **kwargs)
        self.block_initializer = BlockInitialer(
            dynamic_statge_module=GraduallySoft_D_DynamicStaget)

    def set_flop_scale(self, lamda):
        for mut in self.block_mutables:
            mut: GraduallySoft_D_MutableBlocks
            mut.lamda = lamda


@TASK_UTILS.register_module()
class DMSGS_ED_Scheduler(DMSGS_E_Scheduler):
    mutator: DMSGS_D_Mutator

    def before_train_forward(self, iter, epoch, max_iters, max_epochs):
        self.mutator.limit_value()
        if iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            self.mutator.ratio_train()
            self.mutator.set_flop_scale(
                self.current_flop_scale_lam(iter, epoch, max_iters,
                                            max_epochs))
        else:
            self.mutator.requires_grad_(False)

    def current_flop_scale_lam(self, iter, epoch, max_iters, max_epochs):

        if iter < self.decay_ratio * max_iters:
            ratio = epoch / (max_epochs * self.decay_ratio)
        elif iter < (self.decay_ratio + self.refine_ratio) * max_iters:
            ratio = 1.0
        else:
            ratio = 1.0
        return ratio

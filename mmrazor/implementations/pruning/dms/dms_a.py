# dms by channel

# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import MODELS
from ..dtp.modules.dtp_taylor import dtp_get_importance
from .core.mutable import MutableBlocks
from .core.mutator import BlockInitialer, DMSMutator
from .core.op import DynamicStage

# dms adaptive.


class AdaptiveMutableBlocks(MutableBlocks):

    def __init__(self, num_blocks) -> None:
        super().__init__(num_blocks)
        self.taylor = torch.nn.parameter.Parameter(
            torch.zeros([self.num_blocks], requires_grad=True))

    @property
    def current_imp(self):
        imp = dtp_get_importance(self.taylor, self.e, lamda=4.0)

        if self.training and imp.requires_grad:
            with torch.no_grad():
                self.mask.data = (imp >= 0.1).float()
        return imp


class AdaptiveDynamicStaget(DynamicStage):

    def __init__(self, *args):
        super().__init__(*args)

        self.register_mutable_attr(
            'mutable_blocks',
            AdaptiveMutableBlocks(len(list(self.removable_block))))
        self.mutable_blocks: AdaptiveMutableBlocks

        self.prepare_blocks()


@MODELS.register_module()
class DMS_A_Mutator(DMSMutator):

    def __init__(self, *args, **kwargs):
        DMSMutator.__init__(self, *args, **kwargs)
        self.block_initializer = BlockInitialer(
            dynamic_statge_module=AdaptiveDynamicStaget)

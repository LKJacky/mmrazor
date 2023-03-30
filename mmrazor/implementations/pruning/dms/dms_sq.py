# dms by channel

# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import MODELS
from ..dtp.modules.dtp_taylor import dtopk
from .core.mutable import BlockThreshold, MutableBlocks
from .core.mutator import BlockInitialer, DMSMutator
from .core.op import DynamicStage

# dms sequentially.


class SqMutableBlocks(MutableBlocks):

    def __init__(self, num_blocks) -> None:
        super().__init__(num_blocks)
        self.taylor = torch.linspace(0, 1, self.num_blocks)

    @property
    def current_imp(self):
        imp = dtopk(self.taylor, self.e, lamda=4.0)

        if self.training and imp.requires_grad:
            with torch.no_grad():
                self.mask.data = (imp >= BlockThreshold).float()
        return imp


class SqDynamicStaget(DynamicStage):

    def __init__(self, *args):
        super().__init__(*args)

        self.register_mutable_attr(
            'mutable_blocks', SqMutableBlocks(len(list(self.removable_block))))
        self.mutable_blocks: SqMutableBlocks

        self.prepare_blocks()


@MODELS.register_module()
class DMS_Sq_CMutator(DMSMutator):

    def __init__(self, *args, **kwargs):
        DMSMutator.__init__(self, *args, **kwargs)
        self.block_initializer = BlockInitialer(
            dynamic_statge_module=SqDynamicStaget)

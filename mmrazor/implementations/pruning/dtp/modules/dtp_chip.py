# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmrazor.registry import MODELS
from .dtp import DTPUnit


@MODELS.register_module()
class DTPChipUnit(DTPUnit):

    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)

        predefine_imp = torch.zeros([num_channels])
        self.register_buffer('predefine_imp', predefine_imp)
        self.predefine_imp: torch.Tensor

    @torch.no_grad()
    def resort(self):
        imp = self.predefine_imp

        index = imp.sort(descending=True)[1]  # index of big to small
        index_space = torch.linspace(
            0, 1, self.num_channels, device=index.device)  # 0 -> 1
        new_index = torch.zeros_like(imp).scatter(0, index, index_space)
        self.mutable_channel.index.data = new_index

# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops import (DynamicConv2d,
                                                      DynamicLinear)
from .utils import DynamicOpProtocol, replace_with_dynamic_ops


class DistillSparseGptMixin(DynamicOpProtocol):

    def _distill_sparse_gpt_mixin_init(self, *init_args, **int_kwargs):
        self.copy_module: nn.Module = self.static_op_factory(
            *init_args, **int_kwargs)
        self.init_args = init_args
        self.init_kwargs = int_kwargs

        self.loss = None

    def distill(self, y1, y2):
        return (y1 - y2).norm(2)

    def forward(self, x: torch.Tensor):
        x1, x2 = torch.split(x, x.shape[0] // 2, dim=0)

        y1 = self.forward(x1)
        y2 = self.copy_module(x2)
        y = torch.cat([y1, y2], dim=-1)

        if self.training:
            loss = self.distill(y1, y2)
            self.loss = loss
        else:
            self.loss = None
        return y


class DistillSparseGptLinear(DynamicLinear, DistillSparseGptMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._distill_sparse_gpt_mixin_init(*args, **kwargs)

    @classmethod
    def convert_from(cls, module: nn.Linear) -> 'DynamicConv2d':
        new_module = super().convert_from(module)
        new_module.load_state_dict(module.state_dict(), strict=False)

        dtype = next(module.parameters()).dtype
        new_module = new_module.to(dtype)

        return new_module


class DistillSparseGptMutator():

    def __init__(self) -> None:
        self.model: nn.Module = None

    def prepare_from_supernet(self, model: nn.Module) -> None:
        self.model = model
        prune_modules: dict = {}
        prune_modules[nn.Linear] = DistillSparseGptLinear
        replace_with_dynamic_ops(model, prune_modules)

    def gather_distill_loss(self):
        loss = 0
        for op in self.sparse_ops:
            if op.loss is not None:
                loss = loss + op.loss
        return loss

    @property
    def sparse_ops(self):
        assert self.model is not None
        for module in self.model.modules():
            if isinstance(module, DistillSparseGptMixin):
                yield module

    @property
    def named_sparse_ops(self):
        for name, module in self.model.named_modules():
            if isinstance(module, DistillSparseGptMixin):
                yield name, module

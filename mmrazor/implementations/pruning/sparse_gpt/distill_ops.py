# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops import (DynamicConv2d,
                                                      DynamicLinear)
from .utils import DynamicOpProtocol, replace_with_dynamic_ops
import torch.nn.functional as F


class DistillSparseGptMixin(DynamicOpProtocol):

    def _distill_sparse_gpt_mixin_init(self, *init_args, **int_kwargs):
        self.copy_module: nn.Module = self.static_op_factory(
            *init_args, **int_kwargs)
        self.init_args = init_args
        self.init_kwargs = int_kwargs
        self.mask = nn.Parameter(
            torch.ones_like(self.weight), requires_grad=True)

        self.loss = None

    def distill(self, y1, y2):
        return (y1 - y2).norm(2)

    def ditill_forward(self, x: torch.Tensor):
        if self.training:
            self.update_mask()
            from mmrazor.utils import print_log
            print_log(f'{x.shape}')
            x1, x2 = torch.split(x, x.shape[0] // 2, dim=0)

            y1 = self.self_forward(x1)
            y2 = self.copy_module(x2)
            y = torch.cat([y1, y2], dim=0)

            if self.training:
                loss = self.distill(y1, y2)
                self.loss = loss
            else:
                self.loss = None
            print_log
            return y
        else:
            return self.self_forward(x)

    def self_forward(self, x: torch.Tensor):
        raise NotImplementedError()

    @torch.no_grad()
    def update_mask(self):
        weight = self.weight.reshape([-1, 4])
        index = weight.topk(2, dim=-1)[1]  # N 2
        mask = torch.zeros_like(weight)
        mask.scatter_(dim=-1, index=index, value=1)
        self.mask.copy_(mask)


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

    def forward(self, x: torch.Tensor):

        return DistillSparseGptMixin.ditill_forward(self, x)

    def self_forward(self, x: torch.Tensor):
        weight = self.weight * self.mask
        y = F.linear(x, weight, self.bias)
        return y


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

    @torch.no_grad()
    def update_masks(self):
        for op in self.sparse_ops:
            op.update_mask()

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

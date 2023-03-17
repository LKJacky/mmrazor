# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterator

import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops import DynamicMixin
from ...dtp.modules.ops import QuickFlopMixin
from .mutable import MutableBlocks


class DynamicBlockMixin(DynamicMixin, QuickFlopMixin):

    def _dynamic_block_init(self):
        self._scale_func = None
        self._quick_flop_init()

    @property
    def scale(self):
        if self._scale_func is None:
            return 1.0
        else:
            return self._scale_func()

    @property
    def is_removable(self):
        return True

    # inherit from DynamicMixin

    @classmethod
    def convert_from(cls, module):
        pass

    def to_static_op(self) -> nn.Module:
        raise NotImplementedError()

    def register_mutable_attr(self, attr: str, mutable):
        raise NotImplementedError()

    def static_op_factory(self):
        raise NotImplementedError()

    def soft_flop(self: nn.Module):
        flops = 0
        for child in self.children():
            flops = flops + QuickFlopMixin.get_flop(child)
        scale = self.scale
        if isinstance(scale, float):
            scale = 1.0
        else:
            scale = (scale >= 0.5).float().detach() - scale.detach() + scale
        return scale * flops


class DynamicStage(nn.Sequential, DynamicMixin):

    def __init__(self, *args):
        super().__init__(*args)
        self.mutable_attrs = {}
        self.register_mutable_attr(
            'mutable_blocks', MutableBlocks(len(list(self.removable_block))))
        self.mutable_blocks: MutableBlocks

        for i, block in enumerate(self.removable_block):
            block._scale_func = self.mutable_blocks.block_scale_fun_wrapper(i)

    @property
    def removable_block(self) -> Iterator[DynamicBlockMixin]:
        for block in self:
            if isinstance(block, DynamicBlockMixin):
                if block.is_removable:
                    yield block

    @property
    def mutable_blocks(self):
        assert 'mutable_blocks' in self.mutable_attrs
        return self.mutable_attrs['mutable_blocks']

    # inherit from DynamicMixin

    @classmethod
    def convert_from(cls, module):
        return cls(module._modules)

    def to_static_op(self) -> nn.Module:
        modules = []
        i = 0
        for module in self:
            if isinstance(module, DynamicBlockMixin) and module.is_removable:
                if self.mutable_blocks.mask[i] < 0.5:
                    i += 1
                    continue
                else:
                    i += 1
            modules.append(module)
        return nn.Sequential(*modules)

    def static_op_factory(self):
        return super().static_op_factory

    def register_mutable_attr(self, attr: str, mutable):
        self.mutable_attrs[attr] = mutable

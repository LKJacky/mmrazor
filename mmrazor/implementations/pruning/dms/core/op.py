# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterator

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops import DynamicMixin
from ...dtp.modules.ops import QuickFlopMixin
from .mutable import MutableBlocks


class DynamicBlockMixin(DynamicMixin, QuickFlopMixin):

    def _dynamic_block_init(self):
        self._scale_func = None
        self._flop_scale_func = None
        self._quick_flop_init()

    @property
    def scale(self):
        if self._scale_func is None:
            return 1.0
        else:
            scale: torch.Tensor = self._scale_func()
            if scale.numel() == 1:
                return scale
            else:
                return scale.view([1, -1, 1, 1])

    @property
    def flop_scale(self):
        if self._flop_scale_func is None:
            return 1.0
        else:
            scale: torch.Tensor = self._flop_scale_func()
            return scale

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
        scale = self.flop_scale
        return scale * flops

    @property
    def out_channel(self):
        raise NotImplementedError()


class DynamicStage(nn.Sequential, DynamicMixin):

    def __init__(self, *args):
        super().__init__(*args)
        self.mutable_attrs = {}
        self.register_mutable_attr(
            'mutable_blocks', MutableBlocks(len(list(self.removable_block))))
        self.mutable_blocks: MutableBlocks

        for i, block in enumerate(self.removable_block):
            block._scale_func = self.mutable_blocks.block_scale_fun_wrapper(i)
            block._flop_scale_func = self.mutable_blocks.block_flop_scale_fun_wrapper(  # noqa
                i)

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
        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
        modules = []
        i = 0
        for module in self:
            if isinstance(module, DynamicBlockMixin) and module.is_removable:
                if self.mutable_blocks.mask[i] < 0.5:
                    pass
                else:
                    modules.append(module)
                i += 1
            else:
                modules.append(module)

        module = nn.Sequential(*modules)
        return _dynamic_to_static(module)

    def static_op_factory(self):
        return super().static_op_factory

    def register_mutable_attr(self, attr: str, mutable):
        self.mutable_attrs[attr] = mutable

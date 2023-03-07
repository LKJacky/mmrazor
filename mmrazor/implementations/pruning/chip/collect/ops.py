# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from mmrazor.models.architectures.dynamic_ops.bricks.dynamic_conv import \
    DynamicConv2d
from mmrazor.models.architectures.dynamic_ops.bricks.dynamic_linear import \
    DynamicLinear


class CollectMixin:
    """The mixin class for GroupFisher ops."""

    def _collect_init(self) -> None:
        self.handlers: list = []
        self.recorded_input: List = []
        self.recorded_grad: List = []
        self.recorded_out_shape: List = []

    def forward_hook_wrapper(self):
        """Wrap the hook used in forward."""

        def forward_hook(module: CollectMixin, input, output):
            module.recorded_out_shape.append(output.shape)
            module.recorded_input.append(input[0])

        return forward_hook

    def backward_hook_wrapper(self):
        """Wrap the hook used in backward."""

        def backward_hook(module: CollectMixin, grad_in, grad_out):
            module.recorded_grad.insert(0, grad_in[0])

        return backward_hook

    def start_record(self: torch.nn.Module) -> None:
        """Start recording information during forward and backward."""
        self.end_record()  # ensure to run start_record only once
        self.handlers.append(
            self.register_forward_hook(self.forward_hook_wrapper()))
        self.handlers.append(
            self.register_backward_hook(self.backward_hook_wrapper()))

    def end_record(self):
        """Stop recording information during forward and backward."""
        for handle in self.handlers:
            handle.remove()
        self.handlers = []

    def reset_recorded(self):
        """Reset the recorded information."""
        self.recorded_input = []
        self.recorded_grad = []
        self.recorded_out_shape = []


class CollectConv2d(DynamicConv2d, CollectMixin):
    """The Dynamic Conv2d operation used in GroupFisher Algorithm."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._collect_init()


class CollectLinear(DynamicLinear, CollectMixin):
    """The Dynamic Linear operation used in GroupFisher Algorithm."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._collect_init()

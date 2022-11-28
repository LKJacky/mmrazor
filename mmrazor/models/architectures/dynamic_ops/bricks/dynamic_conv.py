# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from ..mixins.dynamic_conv_mixins import (BigNasConvMixin, DynamicConvMixin,
                                          OFAConvMixin)

GroupWiseConvWarned = False


@MODELS.register_module()
class DynamicConv2d(nn.Conv2d, DynamicConvMixin):
    """Dynamic Conv2d OP.

    Note:
        Arguments for ``__init__`` of ``DynamicConv2d`` is totally same as
        :obj:`torch.nn.Conv2d`.

    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `in_channels`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """
    accepted_mutable_attrs = {'in_channels', 'out_channels'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        assert self.padding_mode == 'zeros'
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'DynamicConv2d':
        """Convert an instance of nn.Conv2d to a new instance of
        DynamicConv2d."""

        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True if module.bias is not None else False,
            padding_mode=module.padding_mode)

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return nn.Conv2d

    def forward(self, x: Tensor) -> Tensor:
        """Forward of dynamic conv2d OP."""
        return self.forward_mixin(x)


@MODELS.register_module()
class BigNasConv2d(nn.Conv2d, BigNasConvMixin):
    """Conv2d used in BigNas.

    Note:
        Arguments for ``__init__`` of ``DynamicConv2d`` is totally same as
        :obj:`torch.nn.Conv2d`.

    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `in_channels`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """
    accepted_mutable_attrs = {'in_channels', 'out_channels', 'kernel_size'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        assert self.padding_mode == 'zeros'
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'BigNasConv2d':
        """Convert an instance of `nn.Conv2d` to a new instance of
        `BigNasConv2d`."""
        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True if module.bias is not None else False,
            padding_mode=module.padding_mode)

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return nn.Conv2d

    def forward(self, x: Tensor) -> Tensor:
        """Forward of bignas' conv2d."""
        return self.forward_mixin(x)


@MODELS.register_module()
class OFAConv2d(nn.Conv2d, OFAConvMixin):
    """Conv2d used in `Once-for-All`.

    Refers to `Once-for-All: Train One Network and Specialize it for Efficient
    Deployment <http://arxiv.org/abs/1908.09791>`_.
    """
    """Dynamic Conv2d OP.

    Note:
        Arguments for ``__init__`` of ``OFAConv2d`` is totally same as
        :obj:`torch.nn.Conv2d`.

    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `in_channels`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """
    accepted_mutable_attrs = {'in_channels', 'out_channels'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        assert self.padding_mode == 'zeros'
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'OFAConv2d':
        """Convert an instance of `nn.Conv2d` to a new instance of
        `OFAConv2d`."""

        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True if module.bias is not None else False,
            padding_mode=module.padding_mode)

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return nn.Conv2d

    def forward(self, x: Tensor) -> Tensor:
        """Forward of OFA's conv2d."""
        return self.forward_mixin(x)


@MODELS.register_module()
class DynamicConv2dAdaptivePadding(DynamicConv2d):
    """Dynamic version of mmcv.cnn.bricks.Conv2dAdaptivePadding."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = (
            max((output_h - 1) * self.stride[0] +
                (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0))
        pad_w = (
            max((output_w - 1) * self.stride[1] +
                (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0))
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [
                pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
            ])
        return super().forward(x)


def normal_equation(X: torch.Tensor, Y: torch.Tensor):
    '''
    X: M * N
    Y: M * K
    R: N * K
    X@R=Y
    '''
    origin_x = X
    X = X.double()
    Y = Y.double()
    try:
        R = (X.T @ X).inverse() @ X.T @ Y
    except Exception:
        R = (X.T @ X).pinverse() @ X.T @ Y
    return R.to(origin_x)


@MODELS.register_module()
class LSPDynamicConv2d(DynamicConv2d):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p_weight = self.weight.clone().detach()

    def get_dynamic_params(self):
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Tensor, Optional[Tensor], Tuple[int]]: Sliced weight, bias
                and padding.
        """
        # slice in/out channel of weight according to
        # mutable in_channels/out_channels
        self.p_weight = self.p_weight.to(self.weight)
        weight, bias = self._get_dynamic_params_by_mutable_channels(
            self.p_weight, self.bias)
        return weight, bias, self.padding

    def _get_dynamic_params_by_mutable_channels(self, weight: Tensor, bias):
        if 'in_channels' not in self.mutable_attrs and \
                'out_channels' not in self.mutable_attrs:
            return weight, bias

        if 'out_channels' in self.mutable_attrs:
            mutable_out_channels = self.mutable_attrs['out_channels']
            out_mask = mutable_out_channels.current_mask.to(weight.device)
        else:
            out_mask = torch.ones(weight.size(0)).bool().to(weight.device)

        if self.groups == 1:
            weight = weight[out_mask]
        elif self.groups == self.in_channels == self.out_channels:
            # depth-wise conv
            weight = weight[out_mask]
        else:
            # group-wise conv
            raise NotImplementedError()

        bias = self.bias[out_mask] if self.bias is not None else None
        return weight, bias

    def refresh_weight(self, in_feature=None):

        with torch.no_grad():
            weight = self.p_weight

            if 'in_channels' in self.mutable_attrs:
                mutable_in_channels = self.mutable_attrs['in_channels']
                in_mask = mutable_in_channels.current_mask.to(weight.device)
            else:
                in_mask = torch.ones(weight.size(1)).bool().to(weight.device)

            if self.groups == 1:
                weight = self.get_linear_proj(in_feature, in_mask)
            elif self.groups == self.in_channels == self.out_channels:
                # depth-wise conv
                pass
            else:
                raise NotImplementedError()

            self.p_weight.data = weight
            from mmengine import dist
            dist.broadcast(self.p_weight)

    def get_linear_proj(self, in_feature: torch.Tensor,
                        select_mask: torch.Tensor):
        with torch.no_grad():
            fileted_feature = in_feature[select_mask]
            proj = normal_equation(
                fileted_feature.transpose(-1, -2),
                in_feature.transpose(-1, -2),
            )  # in' in
            proj = proj.T
            weight = self.weight.permute([0, 2, 3, 1]).flatten(0,
                                                               2)  # out in k k
            weight = weight @ proj.to(weight.device)  #
            weight = weight.reshape([
                self.out_channels, self.kernel_size[0], self.kernel_size[1], -1
            ]).permute([0, 3, 1, 2])
            return weight

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        res = super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                            strict, missing_keys,
                                            unexpected_keys, error_msgs)
        self.p_weight.data = self.weight.data
        return res

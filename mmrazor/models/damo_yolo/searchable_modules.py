# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmrazor.models.mutables import SquentialMutableChannel
from mmrazor.registry import MODELS
from .dynamic_utils import (OP_MAPPING, DynamicOneshotModule, MutableDepth,
                            MutableValue, SearchableModuleMinin,
                            replace_with_dynamic_ops)
from .modules import conv_bn, get_activation, get_norm


class SearchableConvBnAct(BaseModule, SearchableModuleMinin):
    """A Conv2d -> Batchnorm -> silu/leaky relu block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride=1,
        groups=1,
        bias=False,
        act='silu',
        norm='bn',
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))

    def init_search_space(self, mutable_in, mutable_out):
        replace_with_dynamic_ops(self)
        if mutable_in is None:
            mutable_in = SquentialMutableChannel(self.conv.in_channels)
        if mutable_out is None:
            mutable_out = SquentialMutableChannel(self.conv.out_channels)

        self.mutable_in = mutable_in
        self.mutable_out = mutable_out

        self.conv.register_mutable_attr('in_channels', mutable_in)
        self.conv.register_mutable_attr('out_channels', mutable_out)

        if self.with_norm:
            self.bn.register_mutable_attr('num_features', mutable_out)

    @classmethod
    def convert_from(cls, module):
        return cls(module.conv.in_channels, module.conv.out_channels,
                   module.conv.kernel_size[0], module.conv.stride,
                   module.conv.groups, module.conv.bias, module.act, module.bn)


@MODELS.register_module()
class SearchableFocus(BaseModule, SearchableModuleMinin):
    """Focus width and height information into channel space."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 act='silu'):
        super().__init__()
        self.conv = SearchableConvBnAct(
            in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

    def init_search_space(self, mutable_in=None, mutable_out=None):
        replace_with_dynamic_ops(self)
        if mutable_in is None:
            mutable_in = SquentialMutableChannel(self.conv.conv.in_channels //
                                                 4)
        if mutable_out is None:
            mutable_out = SquentialMutableChannel(self.conv.conv.out_channels)

        self.mutable_in = mutable_in
        self.mutable_out = mutable_out

        self.conv: SearchableConvBnAct
        self.conv.init_search_space(mutable_in * 4, mutable_out)


class SearchableSPPBottleneck(BaseModule, SearchableModuleMinin):
    """Spatial pyramid pooling layer used in YOLOv3-SPP."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = SearchableConvBnAct(
            in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = SearchableConvBnAct(
            conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

    def init_search_space(self,
                          mutable_in=None,
                          mutable_out=None,
                          mutable_mid=None):
        replace_with_dynamic_ops(self, OP_MAPPING)

        if mutable_in is None:
            mutable_in = SquentialMutableChannel(self.conv1.conv.in_channels)
        if mutable_out is None:
            mutable_out = SquentialMutableChannel(self.conv2.conv.out_channels)
        if mutable_mid is None:
            mutable_mid = SquentialMutableChannel(self.conv1.conv.out_channels)

        self.mutable_in = mutable_in
        self.mutable_out = mutable_out
        self.mutable_mid = mutable_mid

        self.conv1: SearchableConvBnAct
        self.conv2: SearchableConvBnAct

        self.conv1.init_search_space(mutable_in, mutable_mid)
        self.conv2.init_search_space(mutable_mid * (len(self.m) + 1),
                                     mutable_out)


class SearchableRepConv(BaseModule, SearchableModuleMinin):
    """RepConv is a basic rep-style block, including training and deploy status
    Code is based on
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act='relu',
                 norm=None):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if isinstance(act, str):
            self.nonlinearity = get_activation(act)
        else:
            self.nonlinearity = act

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            self.rbr_identity = None
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups)
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups)

    def forward(self, inputs):
        """Forward process."""
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

    def init_search_space(self, mutable_in=None, mutable_out=None):
        # op_mapping = copy.copy(OP_MAPPING)
        # op_mapping.pop(nn.Sequential)

        replace_with_dynamic_ops(self)

        if mutable_in is None:
            mutable_in = SquentialMutableChannel(self.in_channels)
        if mutable_out is None:
            mutable_out = SquentialMutableChannel(self.out_channels)

        self.mutable_in = mutable_in
        self.mutable_out = mutable_out

        if self.deploy:
            self.rbr_reparam.register_mutable_attr('in_channels', mutable_in)
            self.rbr_reparam.register_mutable_attr('out_channels', mutable_out)
        else:
            self.rbr_dense[0].register_mutable_attr('in_channels', mutable_in)
            self.rbr_dense[0].register_mutable_attr('out_channels',
                                                    mutable_out)
            self.rbr_dense[1].register_mutable_attr('num_features',
                                                    mutable_out)

            self.rbr_1x1[0].register_mutable_attr('in_channels', mutable_in)
            self.rbr_1x1[0].register_mutable_attr('out_channels', mutable_out)
            self.rbr_1x1[1].register_mutable_attr('num_features', mutable_out)


class SearchableConvKXBN(BaseModule, SearchableModuleMinin):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 groups=1,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride in [1, 2]

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # lack skip bn
        # lack drop channel
        return out

    def init_search_space(self, mutable_in=None, mutable_out=None):
        replace_with_dynamic_ops(self)

        if mutable_in is None:
            mutable_in = SquentialMutableChannel(self.in_channels)
        if mutable_out is None:
            mutable_out = SquentialMutableChannel(self.out_channels)

        self.mutable_in = mutable_in
        self.mutable_out = mutable_out

        self.conv1.register_mutable_attr('in_channels', mutable_in)
        self.conv1.register_mutable_attr('out_channels', mutable_out)

        self.bn1.register_mutable_attr('num_features', mutable_out)


class SearchableResConvBlock(BaseModule, SearchableModuleMinin):

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=2,
        kernel_size=3,
        padding=None,
        groups=1,
        act='relu',
        reparam=False,
        block_type='k1kx',
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = bottleneck_channels
        self.stride = stride
        assert stride in [1, 2]

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.block_type = block_type
        self.conv1 = DynamicOneshotModule()
        self.mutable_block_type = MutableValue(['k1kx', 'kxkx'])

        self.conv1['k1kx'] = SearchableConvKXBN(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            groups=groups)
        self.conv1['kxkx'] = SearchableConvKXBN(
            in_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=groups)

        if not reparam:
            self.conv2 = SearchableConvKXBN(
                bottleneck_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups)
        else:
            self.conv2 = SearchableRepConv(
                bottleneck_channels,
                out_channels,
                kernel_size,
                stride=stride,
                act='identity')

        if stride != 2:
            self.residual_proj = SearchableConvKXBN(in_channels, out_channels,
                                                    1, 1)
        else:
            self.residual_proj = nn.Identity()

        self.activation_function = get_activation(act)

    def forward(self, x):

        y = self.conv1(x)
        # lack dropout
        y = self.activation_function(y)
        y = self.conv2(y)

        # drop channel

        # drop layer

        if self.stride != 2:
            if y.shape[1] != x.shape[1]:
                y = y + self.residual_proj(x)
            else:
                y = y + x

        y = self.activation_function(y)
        return y

    def init_search_space(
        self,
        mutable_in=None,
        mutable_out=None,
        mutable_mid=None,
        mutable_block_type=None,
    ):
        if mutable_in is None:
            mutable_in = SquentialMutableChannel(self.in_channels)
        if mutable_out is None:
            mutable_out = SquentialMutableChannel(self.out_channels)
        if mutable_mid is None:
            mutable_mid = SquentialMutableChannel(self.bottleneck_channels)
        if mutable_block_type is None:
            mutable_block_type = MutableValue(['k1kx', 'kxkx'])

        self.mutable_in = mutable_in
        self.mutable_out = mutable_out
        self.mutable_mid = mutable_mid
        self.mutable_block_type = mutable_block_type

        for conv in self.conv1.pure_children():
            conv.init_search_space(
                mutable_in=mutable_in, mutable_out=mutable_mid)
        self.conv2.init_search_space(
            mutable_in=mutable_mid, mutable_out=mutable_out)

        self.conv1.register_mutable_attr('module_type', mutable_block_type)

        if isinstance(self.residual_proj, SearchableConvKXBN):
            self.residual_proj.init_search_space(
                mutable_in=mutable_in, mutable_out=mutable_out)


@MODELS.register_module()
class SuperResStem(BaseModule, SearchableModuleMinin):

    def __init__(self, block_setting, depth=1, with_spp=False, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = block_setting['in_channels']
        self.out_channels = block_setting['out_channels']
        self.bottleneck_channels = block_setting['bottleneck_channels']
        self.depth = depth

        block_list = []
        for i in range(depth):
            setting = copy.deepcopy(block_setting)
            if i != 0:
                setting['stride'] = 1
                setting['in_channels'] = setting['out_channels']
            block = SearchableResConvBlock(**setting)
            block_list.append(block)
        if with_spp:
            block_list.insert(
                1,
                SearchableSPPBottleneck(block_setting['out_channels'],
                                        block_setting['out_channels']))
        self.block_list = nn.Sequential(*block_list)

    def forward(self, x):
        return self.block_list(x)

    def init_search_space(self,
                          mutable_in=None,
                          mutable_out=None,
                          mutable_mid=None,
                          mutable_depth=None,
                          mutable_block_type=None):
        if mutable_in is None:
            mutable_in = SquentialMutableChannel(self.in_channels)
        if mutable_out is None:
            mutable_out = SquentialMutableChannel(self.out_channels)
        if mutable_mid is None:
            mutable_mid = SquentialMutableChannel(self.bottleneck_channels)
        if mutable_depth is None:
            mutable_depth = MutableDepth([self.depth // 2, self.depth])
            mutable_depth.current_choice = len(self.block_list)
        if mutable_block_type is None:
            mutable_block_type = MutableValue(['k1kx', 'kxkx'])

        self.mutable_in = mutable_in
        self.mutable_out = mutable_out
        self.mutable_mid = mutable_mid
        self.mutable_depth = mutable_depth
        self.mutable_block_type = mutable_block_type

        replace_with_dynamic_ops(self)

        for i, block in enumerate(self.block_list):
            if isinstance(block, SearchableResConvBlock):
                if i == 0:
                    block.init_search_space(
                        mutable_in=mutable_in,
                        mutable_out=mutable_out,
                        mutable_mid=mutable_mid,
                        mutable_block_type=mutable_block_type)
                else:
                    block.init_search_space(
                        mutable_in=mutable_out,
                        mutable_out=mutable_out,
                        mutable_mid=mutable_mid,
                        mutable_block_type=mutable_block_type)
            elif isinstance(block, SearchableSPPBottleneck):
                block.init_search_space(
                    mutable_in=mutable_out, mutable_out=mutable_out)
        self.block_list.register_mutable_attr('depth', mutable_depth)


@MODELS.register_module()
class TinyNasBackbone(BaseModule, SearchableModuleMinin):

    def __init__(
        self,
        structure_info={},
        out_indices=[2, 4, 5],
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        self.block_list = nn.ModuleList()
        for _, block_info in enumerate(structure_info):
            block = MODELS.build(block_info)
            self.block_list.append(block)

    def forward(self, x):
        ret = []
        for idx, block in enumerate(self.block_list):
            x = block(x)
            if idx in self.out_indices:
                ret.append(x)
        return ret

    def init_search_space(self):
        mutable_block_out_list = nn.ModuleList()
        mutable_out = SquentialMutableChannel(3)
        mutable_out.fix_chosen(mutable_out.current_choice)
        for i, block in enumerate(self.block_list):
            if isinstance(block, SearchableModuleMinin):
                block.init_search_space(mutable_in=mutable_out)
                mutable_out = block.mutable_out
                mutable_block_out_list.append(mutable_out)
        self.mutable_block_out_list = mutable_block_out_list

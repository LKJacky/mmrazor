# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcls.models.backbones.mobilenet_v2 import \
    InvertedResidual as BaseInvertedResidual
from mmcls.models.backbones.mobilenet_v2 import MobileNetV2

from mmrazor.registry import MODELS
from .op import DynamicBlockMixin


class InvertedResidual(BaseInvertedResidual, DynamicBlockMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs
        self._dynamic_block_init()

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x) * self.scale
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

    @property
    def is_removable(self):
        return self.use_res_connect

    def to_static_op(self) -> nn.Module:
        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
        module = BaseInvertedResidual(*self.init_args, **self.init_kwargs)
        for name, m in self.named_children():
            assert hasattr(module, name)
            setattr(module, name, _dynamic_to_static(m))
        return module

    @property
    def out_channel(self):
        raise NotImplementedError()

    def __repr__(self):
        return super().__repr__(
        ) + '\n' + f'res_connect {self.use_res_connect} '


class MobileNetLayers(nn.Sequential):
    pass


@MODELS.register_module()
class DmsMobileNetV2(MobileNetV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        layer, stride
        1,1
        2,2,
        3,2
        4,2
        5,1
        6,2
        7,1
        """
        self.out_indices = (4, )

        def merge(layer1: nn.Sequential, layer2: nn.Sequential):
            return list(layer1._modules.values()) + list(
                layer2._modules.values())

        def re_add(name):
            module = getattr(self, name)
            delattr(self, name)
            return module

        self.conv1 = re_add('conv1')
        self.layer12 = nn.Sequential(*merge(self.layer1, self.layer2))
        self.layer3 = re_add('layer3')
        self.layer45 = nn.Sequential(*merge(self.layer4, self.layer5))
        self.layer67 = nn.Sequential(*merge(self.layer6, self.layer7))
        self.conv2 = re_add('conv2')
        delattr(self, 'layer1')
        delattr(self, 'layer2')
        delattr(self, 'layer4')
        delattr(self, 'layer5')
        delattr(self, 'layer6')
        delattr(self, 'layer7')
        self.layers = ['layer12', 'layer3', 'layer45', 'layer67', 'conv2']

        self.replace_with_mobilenet_layers()

    def replace_with_mobilenet_layers(self):

        def replace(base_layers: nn.Sequential):
            layers = MobileNetLayers()
            for name, module in base_layers._modules.items():
                layers.add_module(name, module)
            return layers

        self.layer12 = replace(self.layer12)
        self.layer3 = replace(self.layer3)
        self.layer45 = replace(self.layer45)
        self.layer67 = replace(self.layer67)

    def convert_checkpoint(self, checkpoint: dict):
        import copy
        checkpoint = copy.deepcopy(checkpoint)

        own_state = self.state_dict()
        new_checkpoint = OrderedDict()
        for key in checkpoint:
            if 'backbone' in key:
                new_checkpoint[key] = checkpoint[key]

        for key1, key2 in zip(own_state, new_checkpoint):
            if own_state[key1].shape == new_checkpoint[key2].shape:
                own_state[key1] = new_checkpoint[key2]
                print(f'match {key1}\t{key2}')
            else:
                print(f'{key1} != {key2}')

        return own_state

    def save_converted_checkpoint(self, checkpoint):
        import copy
        base_checkpoint: dict = checkpoint
        checkpoint = copy.deepcopy(checkpoint)

        own_state = self.state_dict()
        new_checkpoint = OrderedDict()
        for key in checkpoint:
            if 'backbone' in key:
                new_checkpoint[key] = checkpoint[key]

        for key1, key2 in zip(own_state, new_checkpoint):
            if own_state[key1].shape == new_checkpoint[key2].shape:
                base_checkpoint.pop(key2)
                base_checkpoint[f'backbone.{key1}'] = new_checkpoint[key2]
            else:
                print(f'{key1} != {key2}')
        return base_checkpoint

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                InvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride,
                    expand_ratio=expand_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)


@MODELS.register_module()
class DmsMobileNetV2Ex(DmsMobileNetV2):

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride.
    arch_settings = [[1, 16, 2, 1], [6, 24, 4, 2], [6, 32, 6, 2],
                     [6, 64, 8, 2], [6, 96, 6, 1], [6, 160, 6, 2],
                     [6, 320, 2, 1]]

    def __init__(self, *args, **kwargs):
        kwargs['widen_factor'] = 1.5
        super().__init__(*args, **kwargs)

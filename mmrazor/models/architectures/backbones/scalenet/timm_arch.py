# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch.nn as nn
from timm.models.efficientnet import (_create_effnet, decode_arch_def,
                                      resolve_act_layer, resolve_bn_args,
                                      round_channels)
from timm.models.registry import register_model


@register_model
def scale_net_timm(**kwargs):
    variant = 'scalenet'
    pretrained = False

    def get_block(out_c, kernel=3, stride=2, repeat=1, expand=6):
        return f'ir_r{repeat}_k{kernel}_s{stride}_e{expand}_c{out_c}_se0.25'

    scale_net_arch = [
        [get_block(16, kernel=5, stride=1, repeat=1, expand=1)],
        [get_block(32, kernel=5, stride=2, repeat=1)],
        [
            get_block(40, kernel=7, stride=2, repeat=2),
            get_block(40, kernel=5, stride=1, repeat=1),
            get_block(40, kernel=3, stride=1, repeat=1)
        ],
        [get_block(80, kernel=5, stride=2, repeat=2)],
        [get_block(96, kernel=7, stride=1, repeat=2)],
        [
            get_block(192, kernel=7, stride=2, repeat=1),
            get_block(192, kernel=5, stride=1, repeat=1),
            get_block(192, kernel=7, stride=1, repeat=1)
        ],
        [get_block(320, kernel=5, stride=1, repeat=1)],
    ]
    arch_def = scale_net_arch

    round_chs_fn = partial(round_channels, multiplier=1.0, divisor=8)

    model_kwargs = dict(
        block_args=decode_arch_def(
            arch_def, depth_multiplier=1.0, group_size=None),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        norm_layer=kwargs.pop('norm_layer', None)
        or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model

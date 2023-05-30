# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch.nn as nn
from timm.models.efficientnet import (_create_effnet, decode_arch_def,
                                      resolve_act_layer, resolve_bn_args,
                                      round_channels)
from timm.models.registry import register_model


def get_block(out_c, kernel=3, stride=2, repeat=1, expand=6):
    return f'ir_r{repeat}_k{kernel}_s{stride}_e{expand}_c{out_c}_se0.25'


def get_stage(out_c, kernel=[], stride=1, expand=6):
    config = []
    for i, k in enumerate(kernel):
        config.append(
            get_block(
                out_c,
                k,
                stride if i == 0 else 1,
                repeat=1,
                expand=expand if i == 0 else expand))
    return config


def get_arch(index):
    scale_net_arch = {
        0: [
            get_stage(16, kernel=[5], stride=1, expand=1),
            get_stage(32, kernel=[5], stride=2, expand=6),
            get_stage(40, kernel=[7, 7, 5, 3], stride=2, expand=6),
            get_stage(80, kernel=[5, 5], stride=2, expand=6),
            get_stage(96, kernel=[7, 7], stride=1, expand=6),
            get_stage(192, kernel=[7, 5, 7], stride=2, expand=6),
            get_stage(320, kernel=[5], stride=1, expand=6),
        ],
        1: [
            get_stage(17, kernel=[5, 5], stride=1, expand=1),
            get_stage(34, kernel=[5, 5], stride=2, expand=6),
            get_stage(42, kernel=[7, 7, 5, 3, 3], stride=2, expand=6),
            get_stage(84, kernel=[5, 5, 5], stride=2, expand=6),
            get_stage(100, kernel=[7, 7, 7], stride=1, expand=6),
            get_stage(200, kernel=[7, 5, 7, 7], stride=2, expand=6),
            get_stage(333, kernel=[5, 5], stride=1, expand=6),
        ],
        2: [
            get_stage(20, kernel=[5, 5], stride=1, expand=1),
            get_stage(39, kernel=[5, 5], stride=2, expand=6),
            get_stage(48, kernel=[7, 7, 5, 3, 3, 3], stride=2, expand=6),
            get_stage(96, kernel=[5, 5, 5], stride=2, expand=6),
            get_stage(116, kernel=[7, 7, 7], stride=1, expand=6),
            get_stage(231, kernel=[7, 5, 7, 7, 7], stride=2, expand=6),
            get_stage(384, kernel=[5, 5], stride=1, expand=6),
        ],
        3: [
            get_stage(23, kernel=[5, 5], stride=1, expand=1),
            get_stage(45, kernel=[5, 5], stride=2, expand=6),
            get_stage(56, kernel=[7, 7, 5, 3, 3, 3], stride=2, expand=6),
            get_stage(112, kernel=[5, 5, 5], stride=2, expand=6),
            get_stage(135, kernel=[7, 7, 7], stride=1, expand=6),
            get_stage(269, kernel=[7, 5, 7, 7, 7], stride=2, expand=6),
            get_stage(448, kernel=[5, 5], stride=1, expand=6),
        ],
        4: [
            get_stage(25, kernel=[5, 5], stride=1, expand=1),
            get_stage(50, kernel=[5, 5], stride=2, expand=6),
            get_stage(62, kernel=[7, 7, 5, 3, 3, 3, 3], stride=2, expand=6),
            get_stage(123, kernel=[5, 5, 5, 5], stride=2, expand=6),
            get_stage(148, kernel=[7, 7, 7, 7], stride=1, expand=6),
            get_stage(295, kernel=[7, 5, 7, 7, 7], stride=2, expand=6),
            get_stage(491, kernel=[5, 5], stride=1, expand=6),
        ],
        5: [
            get_stage(28, kernel=[5, 5], stride=1, expand=1),
            get_stage(55, kernel=[5, 5], stride=2, expand=6),
            get_stage(68, kernel=[7, 7, 5, 3, 3, 3, 3, 3], stride=2, expand=6),
            get_stage(136, kernel=[5, 5, 5, 5], stride=2, expand=6),
            get_stage(163, kernel=[7, 7, 7, 7], stride=1, expand=6),
            get_stage(325, kernel=[7, 5, 7, 7, 7, 7], stride=2, expand=6),
            get_stage(541, kernel=[5, 5], stride=1, expand=6),
        ],
    }
    stem_size = {0: 32, 1: 34, 2: 39, 3: 45, 4: 50, 5: 55}
    return scale_net_arch[index], stem_size[index]


def _scale_net_timm(index=0, pretrained=False, **kwargs):
    variant = 'scale_net_timm'

    arch_def, stem_size = get_arch(index=index)

    round_chs_fn = partial(round_channels, multiplier=1.0, divisor=8)

    model_kwargs = dict(
        block_args=decode_arch_def(
            arch_def, depth_multiplier=1.0, group_size=None),
        num_features=round_chs_fn(1280),
        stem_size=stem_size,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        norm_layer=kwargs.pop('norm_layer', None)
        or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


@register_model
def scale_net_timm(pretrained=False, **kwargs):
    model = _scale_net_timm(index=0, pretrained=pretrained, **kwargs)
    model.input_size = [3, 224, 224]
    return model


@register_model
def scale_net_timm1(pretrained=False, **kwargs):
    model = _scale_net_timm(index=1, pretrained=pretrained, **kwargs)
    model.input_size = [3, 256, 256]
    return model


@register_model
def scale_net_timm2(pretrained=False, **kwargs):
    model = _scale_net_timm(index=2, pretrained=pretrained, **kwargs)
    model.input_size = [3, 304, 304]
    return model


@register_model
def scale_net_timm3(pretrained=False, **kwargs):
    model = _scale_net_timm(index=3, pretrained=pretrained, **kwargs)
    model.input_size = [3, 354, 354]
    return model


@register_model
def scale_net_timm4(pretrained=False, **kwargs):
    model = _scale_net_timm(index=4, pretrained=pretrained, **kwargs)
    model.input_size = [3, 458, 458]
    return model


@register_model
def scale_net_timm5(pretrained=False, **kwargs):
    model = _scale_net_timm(index=5, pretrained=pretrained, **kwargs)
    model.input_size = [3, 533, 533]
    return model


if __name__ == '__main__':
    import thop
    import torch

    for model in [
            scale_net_timm(),
            scale_net_timm1(),
            scale_net_timm2(),
            scale_net_timm3(),
            scale_net_timm4(),
            scale_net_timm5(),
    ]:
        res = thop.profile(model, (torch.rand(1, *model.input_size), ))
        print(res[0] / 1e6, res[1] / 1e6)
        print()

# sh ./timm_distributed_train.sh 1 data/imagenet_torch --model scale_net_timm -b 128 --sched step --epochs 300 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr 0.08 --warmup-epochs=3 # noqa

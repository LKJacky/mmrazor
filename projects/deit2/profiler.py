import argparse
from timm import create_model
import torch
import ptflops
import torch.nn as nn
from dms_deit import DeitDms, SplitAttention, MultiheadAttention
from mmcls.registry import MODELS
from mmcv.cnn.bricks import Linear as MmcvLinear
from ptflops.pytorch_ops import linear_flops_counter_hook
import copy

model_small = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='deit-small',
        img_size=224,
        patch_size=16),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original')),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
    _scope_='mmcls',
    data_preprocessor=dict(
        num_classes=1000,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
)
model_small_l = copy.deepcopy(model_small)
model_small_l['backbone']['type'] = 'VisionTransformer2'

model_tiny = copy.deepcopy(model_small)
model_tiny['backbone']['arch'] = 'deit-tiny'
model_tiny['head']['in_channels'] = 192


def split_attention_hook(module: SplitAttention, inputs, output):
    head = module.num_heads
    B, N, C = inputs[0].shape
    out_c = output.shape[-1]
    qk_dim = module.q.out_features // head
    v_dim = module.v.out_features // head

    print(qk_dim, v_dim)

    flops = 0

    flops += B * N * C * head * qk_dim * 2  # qk
    flops += B * N * C * head * v_dim * 1  # v

    flops += B * head * N * N * (qk_dim + v_dim)  # attn
    flops += B * N * head * v_dim * out_c  # outproj

    module.__flops__ += flops


def attention_hook(module: MultiheadAttention, inputs, output):
    head = module.num_heads
    B, N, C = inputs[0].shape
    out_c = output.shape[-1]
    head_dim = module.embed_dims // head

    flops = 0

    flops += B * N * C * head * head_dim * 3  # qkv
    flops += B * head * N * N * head_dim * 2  # attn
    flops += B * N * head * head_dim * out_c  # outproj

    module.__flops__ += flops


def load_algo(model: nn.Module, algo_path: str):
    model = DeitDms(model)
    state = torch.load(algo_path, map_location='cpu')['state_dict']
    model.load_state_dict(state)
    model = model.to_static_model()
    return model


def convert_state_dict(state: dict):
    new_state = {}
    for key in state:
        key: str
        if 'qkv' in key and 'weight' in key:
            shape = state[key].shape
            out = shape[-2]
            new_state[key.replace('qkv', 'q')] = state[key][..., :out // 3, :]
            new_state[key.replace('qkv',
                                  'k')] = state[key][...,
                                                     out // 3:out // 3 * 2, :]
            new_state[key.replace('qkv', 'v')] = state[key][..., -out // 3:, :]
        elif 'qkv' in key and 'bias' in key:
            shape = state[key].shape
            out = shape[-1]
            new_state[key.replace('qkv', 'q')] = state[key][:out // 3]
            new_state[key.replace('qkv',
                                  'k')] = state[key][out // 3:out // 3 * 2]
            new_state[key.replace('qkv', 'v')] = state[key][-out // 3:]
        else:
            new_state[key] = state[key]
    return new_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='tiny')
    parser.add_argument('--alg', type=str, default='')
    parser.add_argument('--sub_alg', type=str, default='')
    args = parser.parse_args()

    cfg = {'small': model_small, 'small_l': model_small_l, 'tiny': model_tiny}

    model = MODELS.build(cfg[args.model])

    if args.alg != '':
        model = load_algo(model, args.alg)
        if args.sub_alg != '':
            model = load_algo(model, args.sub_alg)

    res = ptflops.get_model_complexity_info(
        model, (3, 224, 224),
        print_per_layer_stat=False,
        custom_modules_hooks={
            SplitAttention: split_attention_hook,
            MultiheadAttention: attention_hook,
            MmcvLinear: linear_flops_counter_hook
        },
        verbose=False)

    # model = DeitDms(model)
    # print(model.mutator.info())
    print(res)

    # tiny = MODELS.build(cfg["tiny"])
    # model.load_state_dict(convert_state_dict(tiny.state_dict()), strict=True)
    # x=torch.rand([1,3,224,224])
    # y1=model(x)
    # y2=tiny(x)
    # print((y1-y2).abs().max())
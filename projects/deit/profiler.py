import argparse
from timm import create_model
import torch
import ptflops
import torch.nn as nn
from dms_deit import DeitDms, SplitAttention
from mmcls.registry import MODELS

model_dict = dict(
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


def attention_hook(module: SplitAttention, inputs, output):
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
    state = torch.load(algo_path, map_location='cpu')['state_dict']
    model = DeitDms(model)
    model.load_state_dict(state)
    model = model.to_static_model()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--alg', type=str, default='workdirs/prune_deit_s/epoch_30.pth')
    parser.add_argument('--sub_alg', type=str, default='')
    args = parser.parse_args()

    model = MODELS.build(model_dict)

    if args.alg != '':
        model = load_algo(model, args.alg)
        if args.sub_alg != '':
            model = load_algo(model, args.sub_alg)

    res = ptflops.get_model_complexity_info(
        model, (3, 224, 224),
        print_per_layer_stat=False,
        custom_modules_hooks={SplitAttention: attention_hook})

    model = DeitDms(model)
    print(model.mutator.info())
    print(res)
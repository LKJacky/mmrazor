import unittest

import torch
from mmcls.models.utils.attention import WindowMSA
from torchvision.models.swin_transformer import ShiftedWindowAttention

from mmrazor.implementations.pruning.dms.core.swin import (
    DynamicShiftedWindowAttention, DynamicWindowMSA)
from mmrazor.models.mutables import SquentialMutableChannel
from mmrazor.utils import log_tools

log_tools.set_log_level('debug')


class TestAtten(unittest.TestCase):

    def test_attn(self):
        attn = WindowMSA(128, [7, 7], 4)
        x = torch.rand([2 * 7 * 7, 7 * 7, 128])
        y = attn.forward(x)
        print(y.shape)

        dynamic_atten = DynamicWindowMSA.convert_from(attn)
        y2 = dynamic_atten(x)
        print(dynamic_atten)
        self.assertSequenceEqual(y.shape, y2.shape)

        mutable_in = SquentialMutableChannel(128)
        mutable_out = SquentialMutableChannel(128)

        dynamic_atten.register_mutable_attr('in_channels', mutable_in)
        dynamic_atten.register_mutable_attr('out_channels', mutable_out)

        mutable_in.current_choice = 64
        mutable_out.current_choice = 64
        print(dynamic_atten)
        x = torch.rand([2 * 7 * 7, 7 * 7, 64])
        y = dynamic_atten(x)
        self.assertSequenceEqual(x.shape, y.shape)

    def test_torch_attn(self):
        window_size = [7, 7]
        shift_size = [3, 3]
        H = 52
        W = 52

        attn = ShiftedWindowAttention(128, window_size, shift_size, 4)
        x = torch.rand([2, H, W, 128])
        y = attn(x)
        self.assertSequenceEqual(y.shape, [2, H, W, 128])

        dattn = DynamicShiftedWindowAttention.convert_from(attn)
        y1 = dattn(x)
        self.assertTrue((y == y1).all())

    def test_dynamic_torch_attn(self):
        window_size = [7, 7]
        shift_size = [3, 3]
        H = 52
        W = 52

        attn = ShiftedWindowAttention(128, window_size, shift_size, 4)

        dattn = DynamicShiftedWindowAttention.convert_from(attn)

        mutable_in = SquentialMutableChannel(128)
        mutable_out = SquentialMutableChannel(128)

        dattn.register_mutable_attr('in_channels', mutable_in)
        dattn.register_mutable_attr('out_channels', mutable_out)

        mutable_in.current_choice = 64
        mutable_out.current_choice = 72

        x = torch.rand([2, H, W, 64])
        y1 = dattn(x)
        self.assertSequenceEqual(y1.shape, [2, H, W, 72])

    def test_analyer(self):

        from torchvision.models.swin_transformer import swin_t
        model = swin_t()

        from mmrazor.registry import MODELS
        model_dict = model = dict(
            type='ImageClassifier',
            backbone=dict(
                type='mmrazor.TorchSwinBackbone',
                patch_size=[4, 4],
                embed_dim=128,
                depths=[2, 2, 2, 2],
                num_heads=[4, 8, 16, 32],
                window_size=[7, 7],
                stochastic_depth_prob=0.2),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=1024,
                init_cfg=None,
                loss=dict(
                    type='LabelSmoothLoss',
                    label_smooth_val=0.1,
                    mode='original'),
                cal_acc=False),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
                dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
            ],
            train_cfg=dict(augments=[
                dict(type='Mixup', alpha=0.8),
                dict(type='CutMix', alpha=1.0)
            ]),
            _scope_='mmcls')
        model = MODELS.build(model_dict)

        from mmrazor.models.task_modules import ChannelAnalyzer
        a = ChannelAnalyzer(tracer_type='FxTracer').analyze(model)
        print(a.keys(), len(a))

        print(a)

        from mmrazor.models.mutators import ChannelMutator
        mutator = ChannelMutator(
            parse_cfg=dict(
                _scope_='mmrazor',
                type='ChannelAnalyzer',
                demo_input=(1, 3, 224, 224),
                tracer_type='FxTracer'))
        mutator.prepare_from_supernet(model)
        print(model)

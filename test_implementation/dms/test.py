# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmrazor.implementations.pruning.dms.core.models.resnet import (
    ResNetCifarDMS, ResNetCifarSuper)
from mmrazor.implementations.pruning.dms.core.mutator import (BlockInitialer,
                                                              DMSMutator)
from mmrazor.registry import MODELS


class TestDMS(unittest.TestCase):

    def test_r56(self):
        model = ResNetCifarDMS()
        MODEL = dict(
            type='mmcls.ImageClassifier',
            backbone=dict(type='mmrazor.ResNetCifar', num_classes=10),
            head=dict(
                type='mmcls.LinearClsHead',
                num_classes=10,
                in_channels=64,
                loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            ))
        import torch
        x = torch.randn(1, 3, 32, 32)
        model = MODELS.build(MODEL)
        y = model(x)
        print(y.shape)

    def test_initializer(self):
        block_initialer = BlockInitialer()
        model = ResNetCifarDMS()
        static = model.state_dict()

        _ = block_initialer.prepare_from_supernet(model)
        model.load_state_dict(static)

    def test_mutator(self):
        mutator = DMSMutator()
        model = ResNetCifarDMS()
        mutator.prepare_from_supernet(model)
        print(mutator.info())

        mutator.init_quick_flop(model)
        print(mutator.get_soft_flop(model))
        print(mutator)

    def test_model_super(self):
        model = ResNetCifarSuper()
        print(model)

    # def test_scale_polarize(self):
    #     import torch

    #     from mmrazor.implementations.pruning.dms.dms_gsd import scale_polarize # noqa

    #     for lam in [0.5, 1, 5, 10]:
    #         for i in range(10):
    #             a = torch.tensor([i * 0.1])
    #             print(a, scale_polarize(a, lam=lam), lam)

    def test_mobilenet(self):
        from mmrazor.implementations.pruning.dms.core.models.mobilenet import (
            DmsMobileNetV2, MobileNetV2)
        model = DmsMobileNetV2()
        x = torch.rand([1, 3, 224, 224])
        y = model(x)[-1]
        print(y.shape)

        chechpoint = torch.load(
            'mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
        )['state_dict']
        mobilenetv2: MobileNetV2 = MODELS.build(
            dict(
                type='mmcls.ImageClassifier',
                backbone=dict(type='MobileNetV2', widen_factor=1.0),
                neck=dict(type='GlobalAveragePooling'),
                head=dict(
                    type='LinearClsHead',
                    num_classes=1000,
                    in_channels=1280,
                    loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                    topk=(1, 5),
                )))
        mobilenetv2.load_state_dict(chechpoint)
        mobilenetv2.eval()
        model_base = mobilenetv2.backbone
        y = model_base(x)[-1]

        chechpoint = model.convert_checkpoint(chechpoint)
        model.load_state_dict(chechpoint)
        model.eval()
        y1 = model(x)[-1]
        self.assertTrue((y - y1).abs().max() < 1e-5,
                        f'{(y - y1).abs().max()} ')

        converted_checkpoint = model.save_converted_checkpoint(
            mobilenetv2.state_dict())
        torch.save(converted_checkpoint, 'converted_mobilnetv2.pth')

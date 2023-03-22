# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrazor.implementations.pruning.dms.core.mutator import (BlockInitialer,
                                                              DMSMutator)
from mmrazor.implementations.pruning.dms.core.resnet import (ResNetCifarDMS,
                                                             ResNetCifarSuper)
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

    def test_scale_polarize(self):
        import torch

        from mmrazor.implementations.pruning.dms.dms_gsd import scale_polarize

        for lam in [0.5, 1, 5, 10]:
            for i in range(10):
                a = torch.tensor([i * 0.1])
                print(a, scale_polarize(a, lam=lam), lam)

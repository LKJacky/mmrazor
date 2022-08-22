import unittest

import torch
import torchvision

from mmrazor.models.mutators import ChannelUnitMutator
from mmrazor.structures.graph import ModuleGraph
from tests.data.models import LineModel
from .model import ImportanceConv2d, ImpUnit


class TestImpUnit(unittest.TestCase):

    def test_init(self):
        model = torchvision.models.vgg16_bn()
        mutator = ChannelUnitMutator(model, ImpUnit)

        subnet = mutator.sample_structure()
        mutator.apply_structure(subnet)

        x = torch.rand([2, 3, 224, 224])
        y = model(x)
        self.assertEqual(list(y.shape), [2, 1000])

    def test_train(self):
        # model = torchvision.models.vgg16_bn()
        model = LineModel()
        mutator = ChannelUnitMutator(model, ImpUnit)

        model.train()
        # optim = torch.optim.SGD(model.parameters(), 0.1)
        x = torch.rand([2, 3, 224, 224])
        loss = model(x).sum()
        loss.backward()

        for unit in mutator.prunable_units:
            self.assertTrue(unit.mutable.importance.grad is not None)

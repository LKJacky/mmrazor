import unittest

import torch
import torchvision

from mmrazor.models.mutators import ChannelMutator
from tests.data.models import LineModel
from .model import ImpUnit


class TestImpUnit(unittest.TestCase):

    def test_init(self):
        model = torchvision.models.vgg16_bn()
        model = LineModel()
        mutator = ChannelMutator(channel_unit_cfg=ImpUnit)
        # mutator = ChannelMutator()
        mutator.prepare_from_supernet(model)

        subnet = mutator.sample_choices()
        mutator.set_choices(subnet)
        print(model)
        x = torch.rand([2, 3, 224, 224])
        y = model(x)
        self.assertEqual(list(y.shape), [2, 1000])

    def test_train(self):
        # model = torchvision.models.vgg16_bn()
        model = LineModel()
        mutator = ChannelMutator(channel_unit_cfg=ImpUnit)
        mutator.prepare_from_supernet(model)

        model.train()
        optim = torch.optim.SGD(model.parameters(), 0.1)

        x = torch.rand([2, 3, 224, 224])
        loss = model(x).sum()
        optim.zero_grad()
        loss.backward()

        for unit in mutator.mutable_units:
            self.assertTrue(unit.mutable_channel.importance.grad is not None)

        optim.step()

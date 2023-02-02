# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmrazor.models.mutators import ChannelMutator
from projects.cores.expandable_modules.unit import ExpandUnit, expand_model
from ...data.models import MultiConcatModel


class TestExpand(unittest.TestCase):

    def test_expand(self):
        x = torch.rand([1, 3, 224, 224])
        model = MultiConcatModel()
        print(model)
        mutator = ChannelMutator[ExpandUnit](channel_unit_cfg=ExpandUnit)
        mutator.prepare_from_supernet(model)
        print(mutator.choice_template)
        print(model)

        for unit in mutator.mutable_units:
            unit.expand(1)
            print(unit.mutable_channel.mask.shape)
        expand_model(model, zero=True)
        print(model)
        model(x)

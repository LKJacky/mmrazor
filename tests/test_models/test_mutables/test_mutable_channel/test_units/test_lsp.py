# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmrazor.models.mutables.mutable_channel.units.lsp_channel_unit import \
    LSPMutableChannelUnit
from mmrazor.models.mutators import ChannelMutator
from .....data.models import SingleLineModel


class TestLSP(unittest.TestCase):

    def test_lsp_unit(self):
        model = SingleLineModel()
        units = LSPMutableChannelUnit.init_from_prune_tracer(model)
        print(units)

        mutator = ChannelMutator[LSPMutableChannelUnit](
            channel_unit_cfg={
                'type': 'LSPMutableChannelUnit'
            })
        mutator.prepare_from_supernet(model)
        print(mutator.mutable_units)

        for unit in mutator.mutable_units:
            unit.start_colloct_inputs()

        x = torch.rand([2, 3, 224, 224])
        _ = model(x)
        x = torch.rand([2, 3, 224, 224])
        _ = model(x)
        for unit in mutator.mutable_units:
            unit.end_colloct_inputs()

        for unit in mutator.mutable_units:
            print(unit.input_fm.shape)
            unit.compute_imp()
            print(unit.imp)
            print(unit._generate_mask(3))

        mutator.set_choices(mutator.sample_choices())
        x = torch.rand([2, 3, 224, 224])
        _ = model(x)

        mutator.set_choices(mutator.sample_choices())
        print(model)
        x = torch.rand([2, 3, 224, 224])
        _ = model(x)

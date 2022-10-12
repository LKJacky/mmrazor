# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from unittest import TestCase

import torch

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutators.channel_mutator.channel_mutator import \
    is_dynamic_op_for_fx_tracer
from mmrazor.structures.graph import ModuleGraph
from ...data.tracer_passed_models import PassedModelManager

FULL_TEST = os.getenv('FULL_TEST') == 'true'

sys.setrecursionlimit(int(1e8))

DEVICE = torch.device('cpu')


def is_dynamic_op_fx(module, name):
    return isinstance(module, DynamicChannelMixin)


class ToyCNNPseudoLoss:

    def __call__(self, model):
        pseudo_img = torch.rand(2, 3, 16, 16)
        pseudo_output = model(pseudo_img)
        return pseudo_output.sum()


class TestGraph(TestCase):

    def test_init_from_fx_tracer(self) -> None:
        TestData = PassedModelManager.fx_tracer_passed_models()
        for data in TestData:
            with self.subTest(data=data):
                model = data().to(DEVICE)
                graph = ModuleGraph.init_from_fx_tracer(
                    model,
                    dict(
                        type='RazorFxTracer',
                        is_extra_leaf_module=is_dynamic_op_for_fx_tracer,
                        concrete_args=dict(mode='tensor')))

                # check channels
                self._valid_graph(graph)

    def test_init_from_backward_tracer(self) -> None:
        TestData = PassedModelManager.backward_tracer_passed_models()
        for data in TestData:
            with self.subTest(data=data):
                model = data().to(DEVICE)
                model.eval()
                graph = ModuleGraph.init_from_backward_tracer(model)

                # check channels
                self._valid_graph(graph)

    def _valid_graph(self, graph: ModuleGraph):
        try:
            graph.check()
        except Exception as e:
            self.fail(str(e) + '\n' + str(graph))

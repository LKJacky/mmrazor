# Copyright (c) OpenMMLab. All rights reserved.
import sys
from unittest import TestCase

import torch

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutators.channel_mutator.channel_mutator import \
    is_dynamic_op_for_fx_tracer
from mmrazor.structures.graph import ModuleGraph

sys.setrecursionlimit(int(1e8))

DEVICE = torch.device('cpu')


def is_dynamic_op_fx(module, name):
    return isinstance(module, DynamicChannelMixin)


class ToyCNNPseudoLoss:

    def __call__(self, model):
        pseudo_img = torch.rand(2, 3, 16, 16)
        pseudo_output = model(pseudo_img)
        return pseudo_output.sum()


def valid_graph(graph: ModuleGraph):
    try:
        return True, ''
    except Exception as e:
        return False, f'{e},{graph}'


def _test_init_from_fx_tracer(Model):
    model = Model().to(DEVICE)
    print(f'test {Model}')
    graph = ModuleGraph.init_from_fx_tracer(
        model,
        dict(
            type='RazorFxTracer',
            is_extra_leaf_module=is_dynamic_op_for_fx_tracer,
            concrete_args=dict(mode='tensor')))
    # check channels
    return valid_graph(graph)


def _test_init_from_backward_tracer(Model):
    model = Model().to(DEVICE)
    model.eval()
    print(f'test {Model}')
    graph = ModuleGraph.init_from_backward_tracer(model)
    # check channels
    return valid_graph(graph)


class TestGraph(TestCase):
    pass
    # def test_init_from_fx_tracer(self) -> None:
    #     TestData = BackwardPassedModelManager.include_models()
    #     with SetTorchThread(1):
    #         with mp.Pool() as p:
    #             result = p.map(_test_init_from_fx_tracer, TestData)
    #     for res, model in zip(result, TestData):
    #         with self.subTest(model=model):
    #             self.assertTrue(res[0], res[1])

    # def test_init_from_backward_tracer(self) -> None:
    #     TestData = FxPassedModelManager.include_models()
    #     with SetTorchThread(1) as _:
    #         with mp.Pool() as p:
    #             result = p.map(_test_init_from_backward_tracer, TestData)
    #     for res, model in zip(result, TestData):
    #         # test_init_from_backward_tracer(model)
    #         with self.subTest(model=model):
    #             self.assertTrue(res[0], res[1])

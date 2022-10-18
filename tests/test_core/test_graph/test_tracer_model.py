# Copyright (c) OpenMMLab. All rights reserved.
import os
import signal
import time
from contextlib import contextmanager
from functools import partial
from typing import List
from unittest import TestCase

import torch
import torch.multiprocessing as mp

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel.units import \
    SequentialMutableChannelUnit
from mmrazor.models.mutators.channel_mutator.channel_mutator import \
    is_dynamic_op_for_fx_tracer
from mmrazor.structures.graph import ModuleGraph
from ...data.tracer_passed_models import PassedModelManager
from ...utils import SetTorchThread

# sys.setrecursionlimit(int(1e8))

DEVICE = torch.device('cpu')
DEBUG = os.getenv('DEBUG') == 'true'
POOL_SIZE = mp.cpu_count() if not DEBUG else 1


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds, msg='', activated=(not DEBUG)):

    def signal_handler(signum, frame):
        if activated:
            raise TimeoutException(f'{msg} run over {seconds} s!')

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def is_dynamic_op_fx(module, name):
    return isinstance(module, DynamicChannelMixin)


class ToyCNNPseudoLoss:

    def __call__(self, model):
        pseudo_img = torch.rand(2, 3, 16, 16)
        pseudo_output = model(pseudo_img)
        return pseudo_output.sum()


def _test_tracer_2_graph(model, tracer_type='fx'):

    def _test_fx_tracer_2_graph(model):
        graph = ModuleGraph.init_from_fx_tracer(
            model,
            dict(
                type='RazorFxTracer',
                is_extra_leaf_module=is_dynamic_op_for_fx_tracer,
                concrete_args=dict(mode='tensor')))
        return graph

    def _test_backward_tracer_2_graph(model):
        model.eval()
        graph = ModuleGraph.init_from_backward_tracer(model)
        return graph

    if tracer_type == 'fx':
        graph = _test_fx_tracer_2_graph(model)
    else:
        graph = _test_backward_tracer_2_graph(model)
    return graph


def _test_graph2units(graph: ModuleGraph):
    units = SequentialMutableChannelUnit.init_from_graph(graph)
    return units


def _test_units(units: List[SequentialMutableChannelUnit], model):
    for unit in units:
        unit.prepare_for_pruning(model)
    mutable_units = [unit for unit in units if unit.is_mutable]
    assert len(mutable_units) >= 1, \
        'len of mutable units should greater or equal than 0.'
    for unit in mutable_units:
        choice = unit.sample_choice()
        unit.current_choice = choice
        assert abs(unit.current_choice - choice) < 0.1
    x = torch.rand([2, 3, 224, 224]).to(DEVICE)
    y = model(x)
    assert list(y.shape) == [2, 1000]


def _test_a_model(Model, tracer_type='fx'):
    model = Model()
    model.eval()
    print(f'test {Model} using {tracer_type} tracer.')
    try:
        start = time.time()
        with time_limit(20, 'tracer2graph'):
            # trace a model and get graph
            graph = _test_tracer_2_graph(model, tracer_type)
        with time_limit(60, 'graph2units'):
            # graph 2 unit
            units = _test_graph2units(graph)

        with time_limit(20, 'test units'):
            # get unit
            _test_units(units, model)
        print(f'test {Model} successful.')
        return Model, True, '', time.time() - start
    except Exception as e:
        if DEBUG:
            raise e
        else:
            print(f'test {Model} failed.')
            return Model, False, f'{e}', time.time() - start


class TestTraceModel(TestCase):

    def test_init_from_fx_tracer(self) -> None:
        TestData = PassedModelManager.fx_tracer_passed_models()
        with SetTorchThread(1):
            with mp.Pool(POOL_SIZE) as p:
                result = p.map(
                    partial(_test_a_model, tracer_type='fx'), TestData)
        self.report(result, 'fx')

    def test_init_from_backward_tracer(self) -> None:
        TestData = PassedModelManager.backward_tracer_passed_models()
        with SetTorchThread(1) as _:
            with mp.Pool(POOL_SIZE) as p:
                result = p.map(_test_a_model, [TestData, 'backward'])
        self.report(result, 'backward')

    def report(self, result, fx_type='fx'):
        for model, passed, msg, used_time in result:
            print()
            print(f'Test {model} using {fx_type} tracer. ({int(used_time)})')
            if not passed:
                print(f'failed: {msg}')
            self.assertTrue(passed, msg)

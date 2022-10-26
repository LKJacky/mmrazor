# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import signal
import time
from contextlib import contextmanager
from functools import partial
from typing import List
from unittest import TestCase

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel.units import \
    SequentialMutableChannelUnit
from mmrazor.models.task_modules.tracer.backward_tracer import BackwardTracer
from mmrazor.structures.graph import ModuleGraph
from ...data.tracer_passed_models import (PassedModelManager,
                                          backward_passed_library,
                                          fx_passed_library)
from ...utils import SetTorchThread

# sys.setrecursionlimit(int(1e8))

DEVICE = torch.device('cpu')
FULL_TEST = os.getenv('FULL_TEST') == 'true'
try:
    POOL_SIZE = int(os.getenv('POOL_SIZE'))
except Exception:
    POOL_SIZE = mp.cpu_count()

DEBUG = os.getenv('DEBUG') == 'true'
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)

print(f'FULL_TEST: {FULL_TEST}')
print(f'POOL_SIZE: {POOL_SIZE}')
print(f'DEBUG: {DEBUG}')


class TimeoutException(Exception):
    pass


def get_shape(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.shape
    elif isinstance(tensor, list) or isinstance(tensor, tuple):
        shapes = []
        for x in tensor:
            shapes.append(get_shape(x))
        return shapes
    elif isinstance(tensor, dict):
        shapes = {}
        for key in tensor:
            shapes[key] = get_shape[tensor[key]]
        return shapes
    else:
        raise NotImplementedError(
            f'unsuppored type{type(tensor)} to get shape of tensors.')


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
    from mmcv.cnn.bricks import Scale

    is_leaf = (
        isinstance(module, DynamicChannelMixin)
        or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        or isinstance(module, nn.modules.batchnorm._BatchNorm)
        or isinstance(module, Scale))

    return is_leaf


class SumLoss:

    def __call__(self, model):
        img = torch.zeros([2, 3, 224, 224])
        y = model(img)
        return self.get_loss(y)

    def get_loss(self, output):
        if isinstance(output, torch.Tensor):
            return output.sum()
        elif isinstance(output, list) or isinstance(output, tuple):
            loss = 0
            for out in output:
                loss += self.get_loss(out)
            return loss
        elif isinstance(output, dict):
            return self.get_loss(list(output.values()))
        else:
            raise NotImplementedError(f'{type(output)}')


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
                is_extra_leaf_module=is_dynamic_op_fx,
                concrete_args=dict(mode='tensor')))
        return graph

    def _test_backward_tracer_2_graph(model):
        model.eval()
        graph = ModuleGraph.init_from_backward_tracer(
            model, backward_tracer=BackwardTracer(SumLoss()))
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
    x = torch.rand([2, 3, 224, 224]).to(DEVICE)
    tensors_org = model(x)
    # prune
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
    tensors = model(x)
    assert get_shape(tensors_org) == get_shape(tensors)
    return mutable_units


def _test_a_model(Model, tracer_type='fx'):
    start = time.time()

    try:
        Model.init_model()
        model = Model
        model.eval()
        print(f'test {Model} using {tracer_type} tracer.')

        with time_limit(20, 'tracer2graph'):
            # trace a model and get graph
            graph: ModuleGraph = _test_tracer_2_graph(model, tracer_type)
            num = len(graph)
        with time_limit(120, 'graph2units'):
            # graph 2 unit
            units = _test_graph2units(graph)
            num = len(units)

        with time_limit(30, 'test units'):
            # get unit
            mutable_units = _test_units(units, model)
            num = len(mutable_units)
        print(f'test {Model} successful.')
        return Model, True, '', time.time() - start, num
    except Exception as e:
        if DEBUG:
            raise e
        else:
            print(f'test {Model} failed.')
            return Model, False, f'{e}', time.time() - start, -1


class TestTraceModel(TestCase):

    def test_init_from_fx_tracer(self) -> None:
        TestData = fx_passed_library.include_models(FULL_TEST)
        with SetTorchThread(1):
            with mp.Pool(POOL_SIZE) as p:
                result = p.map(
                    partial(_test_a_model, tracer_type='fx'), TestData)
        self.report(result, fx_passed_library, 'fx')

    def test_init_from_backward_tracer(self) -> None:
        TestData = backward_passed_library.include_models(FULL_TEST)
        with SetTorchThread(1) as _:
            with mp.Pool(POOL_SIZE) as p:
                result = p.map(
                    partial(_test_a_model, tracer_type='backward'), TestData)
        self.report(result, backward_passed_library, 'backward')

    def report(self, result, model_manager: PassedModelManager, fx_type='fx'):
        print()
        print(f'Trace model summary using {fx_type} tracer.')

        passd_test = [res for res in result if res[1] is True]
        unpassd_test = [res for res in result if res[1] is False]

        print(f'{len(passd_test)},{len(unpassd_test)},'
              f'{len(model_manager.uninclude_models(full_test=FULL_TEST))}')

        print('Passed:')
        for model, passed, msg, used_time, len_mutable in passd_test:
            with self.subTest(model=model):
                print(f'\t{model}\t{int(used_time)}s\t{len_mutable}')
                self.assertTrue(passed, msg)

        print('UnPassed:')
        for model, passed, msg, used_time, len_mutable in unpassd_test:
            with self.subTest(model=model):
                print(f'\t{model}\t{int(used_time)}s\t{len_mutable}')
                print(f'\t\t{msg}')
                self.assertTrue(passed, msg)

        print('UnTest:')
        for model in model_manager.uninclude_models(full_test=FULL_TEST):
            print(f'\t{model}')

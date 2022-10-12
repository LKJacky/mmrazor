# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from unittest import TestCase

import torch

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutators.channel_mutator.channel_mutator import \
    is_dynamic_op_for_fx_tracer
from mmrazor.structures.graph import ModuleGraph
from ...data.model_library import MMModelLibrary, TorchModelLibrary
from ...data.models import Icep  # noqa
from ...data.models import MultipleUseModel  # noqa
from ...data.models import Xmodel  # noqa
from ...data.models import (AddCatModel, ConcatModel, ConvAttnModel,
                            DwConvModel, ExpandLineModel, GroupWiseConvModel,
                            LineModel, MultiBindModel, MultiConcatModel,
                            MultiConcatModel2, ResBlock)

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

    @classmethod
    def fx_passed_models(cls):
        default_models = [
            LineModel,
            ResBlock,
            AddCatModel,
            ConcatModel,
            MultiConcatModel,
            MultiConcatModel2,
            GroupWiseConvModel,
            Xmodel,
            MultipleUseModel,
            Icep,
            ExpandLineModel,
            MultiBindModel,
            DwConvModel,  #
            ConvAttnModel,
        ]
        """
        googlenet: return a tuple when training, so it should
        trace in eval mode
        """
        torch_models_includes = [
            'alexnet',
            'densenet',
            'efficientnet',
            'googlenet',
            'inception',
            'mnasnet',
            'mobilenet',
            'regnet',
            'resnet',
            'resnext',
            # 'shufflenet', # bug
            'squeezenet',
            'vgg',
            'wide_resnet',
        ]
        torch_model_library = TorchModelLibrary(include=torch_models_includes)
        """
        shufflenet consists of chunk operations.
        resnest: resnest has two problems. First it uses *x.shape() which is
            not tracerable using fx tracer. Second, it uses channel folding.
        res2net: res2net consists of split operations.
        convnext: consist of layernorm.
        """
        mmcls_model_include = [
            'vgg',
            'efficientnet',
            'resnet',
            'mobilenet',
            'resnext',
            'wide-resnet',
            # 'shufflenet', # bug
            'hrnet',
            # 'resnest',  # bug
            'inception',
            # 'res2net',  # bug
            'densenet',
            # 'convnext',  # bug
            'regnet',
            # transformer and mlp
            # # 'van', # bug
            # # 'swin_transformer', # bug
            # 'convmixer', # bug
            # # 't2t', # bug
            # # 'twins', # bug
            # # 'repmlp', # bug
            # # 'tnt', # bug
            # # 't2t', # bug
            # # 'mlp_mixer', # bug
            # # 'conformer', # bug
            # # 'poolformer', # bug
            # # 'vit', # bug
        ]
        mmcls_exclude = ['cutmix', 'cifar', 'gem']
        mmcls_model_library = MMModelLibrary(
            include=mmcls_model_include, exclude=mmcls_exclude)

        models = default_models \
            + torch_model_library.include_models()\
            + mmcls_model_library.include_models() \
            if FULL_TEST else default_models

        return models

    @classmethod
    def backward_tracer_passed_models(cls):
        '''MultipleUseModel: backward tracer can't distinguish multiple use and
        first bind then use.'''
        default_models = [
            LineModel,
            ResBlock,
            AddCatModel,
            ConcatModel,
            MultiConcatModel,
            MultiConcatModel2,
            GroupWiseConvModel,
            Xmodel,
            # MultipleUseModel,  # bug
            Icep,
            ExpandLineModel,
            MultiBindModel,
            DwConvModel
        ]
        """
        googlenet return a tuple when training, so it
            should trace in eval mode
        """

        torch_models_includes = [
            'alexnet',
            'densenet',
            'efficientnet',
            'googlenet',
            'inception',
            'mnasnet',
            'mobilenet',
            'regnet',
            'resnet',
            'resnext',
            # 'shufflenet',     # bug
            'squeezenet',
            'vgg',
            'wide_resnet',
        ]
        torch_model_library = TorchModelLibrary(include=torch_models_includes)
        """
        shufflenet consists of chunk operations.
        resnest: resnest has two problems. First it uses *x.shape() which is
            not tracerable using fx tracer. Second, it uses channel folding.
        res2net: res2net consists of split operations.
        convnext: consist of layernorm.
        """
        mmcls_model_include = [
            'vgg',
            'efficientnet',
            'resnet',
            'mobilenet',
            'resnext',
            'wide-resnet',
            # 'shufflenet',  # bug
            'hrnet',
            # 'resnest',  # bug
            'inception',
            # 'res2net',  # bug
            'densenet',
            # 'convnext',  # bug
            'regnet',
            # 'van',  # bug
            # 'swin_transformer',  # bug
            # 'convmixer', # bug
            # 't2t',  # bug
            # 'twins',  # bug
            # 'repmlp',  # bug
            # 'tnt',  # bug
            # 't2t',  # bug
            # 'mlp_mixer',  # bug
            # 'conformer',  # bug
            # 'poolformer',  # bug
            # 'vit',  # bug
        ]
        mmcls_exclude = ['cutmix', 'cifar', 'gem']
        mmcls_model_library = MMModelLibrary(
            include=mmcls_model_include, exclude=mmcls_exclude)

        models = default_models \
            + torch_model_library.include_models()\
            + mmcls_model_library.include_models() \
            if FULL_TEST else default_models

        return models

    def test_init_from_fx_tracer(self) -> None:
        TestData = self.fx_passed_models()
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
        TestData = self.backward_tracer_passed_models()

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

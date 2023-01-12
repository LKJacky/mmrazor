# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest

import torch
import torch.nn as nn
from mmengine import fileio

import mmrazor.models.architectures.dynamic_ops as dynamic_ops
from mmrazor.models.damo_yolo.damo_mutator import DamoMutator
from mmrazor.models.damo_yolo.dynamic_utils import (
    DynamicOneshotModule, MutableValue, SearchAableModelDeployWrapper)
from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                  load_fix_subnet)

from mmrazor.models.damo_yolo.searchable_modules import (  # isort:skip
    SearchableResConvBlock, SearchableFocus, SearchableModuleMinin,
    SearchableRepConv, SearchableSPPBottleneck, SuperResStem, TinyNasBackbone)

tmp_folder = os.path.dirname(__file__)
SUPERNET = [
    dict(
        type='SearchableFocus',
        in_channels=3,
        out_channels=1024,
        ksize=3,
        stride=1,
        act='relu',
    ),
    dict(
        type='SuperResStem',
        block_setting=dict(
            in_channels=1024,
            out_channels=1024,
            bottleneck_channels=1024,
            reparam=True,
            block_type='k1kx'),
        depth=5,
    ),
    dict(
        type='SuperResStem',
        block_setting=dict(
            in_channels=1024,
            out_channels=1024,
            bottleneck_channels=1024,
            reparam=True,
            block_type='k1kx'),
        depth=5,
    ),
    dict(
        type='SuperResStem',
        block_setting=dict(
            in_channels=1024,
            out_channels=1024,
            bottleneck_channels=1024,
            reparam=True,
            block_type='k1kx'),
        depth=5,
    ),
    dict(
        type='SuperResStem',
        block_setting=dict(
            in_channels=1024,
            out_channels=1024,
            bottleneck_channels=1024,
            reparam=True,
            stride=1,
            block_type='k1kx'),
        depth=5,
    ),
    dict(
        type='SuperResStem',
        block_setting=dict(
            in_channels=1024,
            out_channels=1024,
            bottleneck_channels=1024,
            reparam=True,
            block_type='k1kx'),
        depth=5,
        with_spp=True,
    )
]


def check_no_dynamic_op(module: nn.Module):
    for m in module.modules():
        if isinstance(m, dynamic_ops.DynamicMixin):
            return False
    return True


class TestDamoYolo(unittest.TestCase):

    def _test_a_searchable(self, module: SearchableModuleMinin, in_c=32):
        x = torch.rand(1, in_c, 128, 128)
        module(x)

        module.init_search_space()
        module(x)

        mutator = DamoMutator()
        mutator.prepare_for_supernet(module)
        if hasattr(module, 'mutable_in'):
            module.mutable_in.fix_chosen()
        mutator.sample_subnet()
        print(module)
        module(x)

        search_space_dict = module.dump()
        mutator.sample_subnet()
        module.load(search_space_dict)
        search_space_dict2 = module.dump()
        self.assertDictEqual(search_space_dict, search_space_dict2)

        module = module.to_static_op()
        module(x)
        self.assertTrue(check_no_dynamic_op(module))

    def test_focus(self):
        layer = SearchableFocus(32, 64)
        self._test_a_searchable(layer)

    def test_spp(self):
        layer = SearchableSPPBottleneck(32, 64)
        self._test_a_searchable(layer)

    def test_rep_conv(self):
        layer = SearchableRepConv(32, 64)
        self._test_a_searchable(layer)

    def test_res_conv_block(self):
        layer = SearchableResConvBlock(
            32, 64, 128, stride=2, kernel_size=3, groups=1)
        self._test_a_searchable(layer)

    def test_super_res_stem(self):
        blocks = SuperResStem(
            block_setting=dict(
                in_channels=32,
                out_channels=64,
                bottleneck_channels=128,
                stride=2,
                kernel_size=3,
                padding=None,
                groups=1,
                act='relu',
            ),
            depth=3)
        self._test_a_searchable(blocks)

    def test_damo_yolo_backbone(self):
        backbone = TinyNasBackbone(SUPERNET)
        self._test_a_searchable(backbone, 3)

    def test_load_and_export(self):
        backbone = TinyNasBackbone(SUPERNET)
        backbone.init_search_space()
        print(backbone)
        subnet = export_fix_subnet(backbone)
        load_fix_subnet(backbone, subnet[0])

    def test_wrapper(self):
        backbone = TinyNasBackbone(SUPERNET)
        backbone.init_search_space()
        backbone = SearchAableModelDeployWrapper(backbone, to_static=True)

        backbone = TinyNasBackbone(SUPERNET)
        backbone.init_search_space()
        mutator = DamoMutator()
        mutator.prepare_for_supernet(backbone)
        mutator.sample_subnet()
        subnet_path = tmp_folder + '/subnet.yaml'
        fileio.dump(backbone.dump(), subnet_path)
        backbone = SearchAableModelDeployWrapper(
            backbone, subnet_dict=subnet_path, to_static=True)
        os.remove(subnet_path)


class TestSearchableOneShotModule(unittest.TestCase):

    def test_one_shot_module(self):
        layer = DynamicOneshotModule()
        layer['a'] = nn.Conv2d(32, 64, 1)
        layer['b'] = nn.Conv2d(32, 128, 1)

        x = torch.rand([1, 32, 64, 64])
        y = layer(x)
        self.assertEqual(list(y.shape), [1, 64, 64, 64])

        mutable = MutableValue(['a', 'b'])
        mutable.current_choice = 'a'
        layer.register_mutable_attr('module_type', mutable)
        y = layer(x)
        self.assertEqual(list(y.shape), [1, 64, 64, 64])

        mutable.current_choice = 'b'
        y = layer(x)
        self.assertEqual(list(y.shape), [1, 128, 64, 64])

        static_op = layer.to_static_op()
        print(static_op)

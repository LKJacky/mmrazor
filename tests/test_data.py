# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from .data.model_library import (DefaultModelLibrary, MMClsModelLibrary,
                                 MMDetModelLibrary, MMSegModelLibrary,
                                 TorchModelLibrary)
from .data.tracer_passed_models import (BackwardPassedModelManager,
                                        FxPassedModelManager)


class TestModelLibrary(unittest.TestCase):

    def test_mmcls(self):
        library = MMClsModelLibrary(exclude=['cutmax', 'cifar'])
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_defaul_library(self):
        library = DefaultModelLibrary()
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_torchlibrary(self):
        library = TorchModelLibrary()
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_mmdet(self):
        library = MMDetModelLibrary()
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_mmseg(self):
        library = MMSegModelLibrary()
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_passed_models(self):
        try:
            print(FxPassedModelManager.include_models(True))
            print(BackwardPassedModelManager.include_models(True))
        except Exception:
            self.fail()


class TestModels(unittest.TestCase):

    def _test_a_model(self, Model):
        model = Model()
        x = torch.rand(2, 3, 224, 224)
        y = model(x)
        self.assertSequenceEqual(y.shape, [2, 1000])

    def test_models(self):
        library = DefaultModelLibrary()
        for Model in library.include_models():
            with self.subTest(model=Model):
                self._test_a_model(Model)

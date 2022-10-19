# Copyright (c) OpenMMLab. All rights reserved.
import unittest

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
        print(library.short_names())
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

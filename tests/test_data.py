# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from .data.model_library import (MMClsModelLibrary, MMDetModelLibrary,
                                 TorchModelLibrary)


class TestModelLibrary(unittest.TestCase):

    def test_mmcls(self):
        library = MMClsModelLibrary(exclude=['cutmax', 'cifar'])
        self.assertTrue(library.is_default_includes_cover_all_models())
        print(library.include_models())

    def test_torchlibrary(self):
        library = TorchModelLibrary()
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_mmdet(self):
        library = MMDetModelLibrary()
        self.assertTrue(library.is_default_includes_cover_all_models())

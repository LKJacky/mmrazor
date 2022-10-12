# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from .data.model_library import MMModelLibrary, TorchModelLibrary


class TestModelLibrary(unittest.TestCase):

    def test_mmlibrary_init(self):
        library = MMModelLibrary(exclude=['cutmax', 'cifar'])
        library.is_default_includes_cover_all_models()

    def test_torchlibrary(self):
        library = TorchModelLibrary()
        library.is_default_includes_cover_all_models()

# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_paths = [
    './docs/pruning/READMEs/mutable_channel_unit.ipynb',
    './docs/pruning/READMEs/channel_mutator.ipynb'
]


class TestDocs(TestCase):

    def test_notebooks(self):
        for path in notebook_paths:
            with self.subTest(path=path):
                with open(path) as file:
                    nb_in = nbformat.read(file, nbformat.NO_CONVERT)
                    ep = ExecutePreprocessor(
                        timeout=600, kernel_name='python3')
                    try:
                        _ = ep.preprocess(nb_in)
                    except Exception:
                        self.fail()

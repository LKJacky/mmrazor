# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from ..modules.mutable_channels import dtopk


class TestDtp(TestCase):

    def test_dtopk(self):
        a = torch.linspace(0, 1, 10)
        b = dtopk(a, torch.tensor(0.5))
        print(b)
        self.assertTrue(b[0] > b[-1])

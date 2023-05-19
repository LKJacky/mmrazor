# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModel

from mmrazor.registry import MODELS
from .op import HSwish, InvertedResidual, ScaleConv2d

stage_cfg = [
    dict(n=1, stride=1, out_c=16, kernel=5, expand_ratio=1),  # 1
    dict(n=1, stride=2, out_c=32, kernel=5, expand_ratio=6),  # 2
    dict(n=4, stride=2, out_c=40, kernel=[7, 7, 5, 3], expand_ratio=6),  # 3
    dict(n=2, stride=2, out_c=80, kernel=5, expand_ratio=6),  # 4
    dict(n=2, stride=1, out_c=96, kernel=7, expand_ratio=6),  # 5
    dict(n=3, stride=2, out_c=192, kernel=[7, 5, 7], expand_ratio=6),  # 6
    dict(n=1, stride=1, out_c=320, kernel=5, expand_ratio=6),  # 7
]


@MODELS.register_module()
class ScaleNet(BaseModel):

    def __init__(self,
                 stem_out=32,
                 stage_cfg=stage_cfg,
                 out_dim=1280,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.conv_stem = self.make_stem(3, stem_out, stride=2, kernel=3)
        self.stages = nn.ModuleList()

        stage_inc = stem_out
        for cfg in stage_cfg:
            self.stages.append(self.make_stage(in_c=stage_inc, **cfg))
            stage_inc = cfg['out_c']

        self.conv_out = self.make_stem(stage_inc, out_dim, stride=1, kernel=1)

    def make_stem(self, in_c, out_c, stride=1, kernel=3):
        return ScaleConv2d(
            inp=in_c, oup=out_c, stride=stride, k=kernel, activation=HSwish)

    def make_stage(self,
                   n=1,
                   stride=1,
                   in_c=32,
                   out_c=32,
                   kernel=3,
                   expand_ratio=1):
        stage = nn.Sequential()
        for i in range(n):
            if isinstance(kernel, int):
                block_kernel = kernel
            else:
                block_kernel = kernel[i]
            module = InvertedResidual(
                inp=in_c if i == 0 else out_c,
                oup=out_c,
                t=expand_ratio,
                stride=stride if i == 0 else 1,
                k=block_kernel,
                n=n,
                activation=HSwish,
                use_se=True)
            stage.append(module)
        return stage

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        x = self.conv_stem(inputs)
        for stage in self.stages:
            x = stage(x)
        x = self.conv_out(x)
        return (x, )

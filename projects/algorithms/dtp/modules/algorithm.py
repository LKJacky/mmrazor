# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmrazor.models import BaseAlgorithm
from mmrazor.registry import MODELS
from mmrazor.utils import RuntimeInfo, print_log
from .mutator import ImpMutator


@MODELS.register_module()
class DTPAlgorithm(BaseAlgorithm):

    def __init__(
            self,
            architecture: Union[BaseModel, Dict],
            mutator_cfg=dict(
                type='ImpMutator',
                channel_unit_cfg=dict(
                    type='ImpUnit',
                    default_args=dict(
                        imp_type='dtp',
                        grad_clip=-1,
                    )),
                parse_cfg=dict(
                    _scope_='mmrazor',
                    type='ChannelAnalyzer',
                    demo_input=dict(
                        type='DefaultDemoInput',
                        input_shape=[1, 3, 224, 224],
                    ),
                    tracer_type='FxTracer'),
            ),
            target_flop=0.5,
            flop_loss_weight=1.0,
            #
            data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
            init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.target_flop = target_flop
        self.flop_loss_weight = flop_loss_weight

        self.mutator: ImpMutator = MODELS.build(mutator_cfg)
        self.mutator.prepare_from_supernet(self.architecture)

        self.original_flops = self.mutator.get_soft_flop(
            self.architecture).detach().item()
        print_log(f'Get init flops {self.original_flops/1e6}')

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor'):
        # update

        if self.training:
            if RuntimeInfo.iter() == 0:
                self.mutator.resort()
                print_log('Resorted')

        res: dict = super().forward(inputs, data_samples, mode)  # type: ignore

        if self.training and mode == 'loss':
            # flop_loss
            current_flops = self.mutator.get_soft_flop(self.architecture)
            res['flop_loss'] = self.flop_loss(
                current_flops) * self.flop_loss_weight
            res['soft_flop'] = current_flops.detach()

        return res

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper) -> Dict[str, torch.Tensor]:

        res = super().train_step(data, optim_wrapper)
        self._train_step(self)
        return res

    def _train_step(self, algorithm):
        algorithm.mutator.limit_value()

    def flop_loss(self, current_flop):
        return (current_flop / self.original_flops - self.target_flop)**2

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmrazor.models import BaseAlgorithm
from mmrazor.registry import MODELS, TASK_UTILS
from mmrazor.utils import RuntimeInfo
from .mutator import DMSMutator
from .scheduler import DMSScheduler


def convert_float_to_tenosr(res: dict, device):
    for k in res:
        if not isinstance(res[k], torch.Tensor):
            res[k] = torch.tensor(res[k], device=device)
    return res


@MODELS.register_module()
class BaseDTPAlgorithm(BaseAlgorithm):

    def __init__(
            self,
            architecture: Union[BaseModel, Dict],
            mutator_cfg=dict(
                type='DMSMutator',
                channel_unit_cfg=dict(type='ImpUnit', default_args=dict()),
                parse_cfg=dict(
                    _scope_='mmrazor',
                    type='ChannelAnalyzer',
                    demo_input=dict(
                        type='DefaultDemoInput',
                        input_shape=[1, 3, 224, 224],
                    ),
                    tracer_type='FxTracer'),
            ),
            scheduler=dict(
                type='',
                flops_target=0.5,
                decay_ratio=0.6,
                refine_ratio=0.2,
                flop_loss_weight=1,
            ),
            #
            data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
            init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.mutator: DMSMutator = MODELS.build(mutator_cfg)
        scheduler.update(dict(mutator=self.mutator, model=self.architecture))
        self.scheduler: DMSScheduler = TASK_UTILS.build(scheduler)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor'):
        if self.training and mode == 'loss':
            self.scheduler.before_train_forward(RuntimeInfo.iter(),
                                                RuntimeInfo.epoch(),
                                                RuntimeInfo().max_iters(),
                                                RuntimeInfo().max_epochs())
        res: dict = super().forward(inputs, data_samples, mode)  # type: ignore
        if self.training and mode == 'loss':
            extra_dict = self.scheduler.after_train_forward(
                RuntimeInfo.iter(), RuntimeInfo.epoch(),
                RuntimeInfo().max_iters(),
                RuntimeInfo().max_epochs())
            extra_dict = convert_float_to_tenosr(extra_dict, inputs.device)
            res.update(extra_dict)
        return res

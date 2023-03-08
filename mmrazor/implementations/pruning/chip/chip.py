# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from torch import distributed as dist

from mmrazor.models import BaseAlgorithm
from mmrazor.registry import MODELS
from mmrazor.utils import print_log
from .collect.mutator import BaseCollectMutator
from .collect.ops import CollectConv2d, CollectLinear
from .collect.unit import BaseCollectUni
from .modules import compute_ci


@MODELS.register_module()
class ChipUnit(BaseCollectUni):

    def __init__(self, num_channels: int, *args) -> None:
        super().__init__(num_channels, *args)

        predefine_imp = torch.zeros([num_channels])
        self.register_buffer('predefine_imp', predefine_imp)
        self.predefine_imp: torch.Tensor

    @torch.no_grad()
    def update_predefine_imp(self):
        new_ci = 0
        for layer in self.input_related_dynamic_ops:
            if isinstance(layer, CollectConv2d):
                tensor = layer.recorded_input[0].flatten(2)
                new_ci += compute_ci(tensor)
            elif isinstance(layer, CollectLinear):
                tensor = layer.recorded_input[0].unsqueeze(-1)
                new_ci += compute_ci(tensor)
            else:
                raise NotImplementedError()
        assert isinstance(new_ci, torch.Tensor)
        new_ci / len(list(self.input_related_dynamic_ops))
        self.predefine_imp.data += new_ci
        if dist.is_initialized():
            dist.all_reduce(self.predefine_imp)

    def info(self):
        return f'{self.name}:\t{self.predefine_imp.min().item()}\t{self.predefine_imp.max().item()}'  # noqa


@MODELS.register_module()
class ChipMutator(BaseCollectMutator):

    def __init__(self,
                 channel_unit_cfg=ChipUnit,
                 parse_cfg=dict(
                     type='ChannelAnalyzer',
                     demo_input=(1, 3, 224, 224),
                     tracer_type='FxTracer'),
                 **kwargs) -> None:
        super().__init__(channel_unit_cfg, parse_cfg, **kwargs)
        self.mutable_units: List[ChipUnit]

    def update_predefine_imp(self) -> None:
        for unit in self.mutable_units:
            print_log(f'{unit.name} update imp')
            unit.update_predefine_imp()
            print_log(unit.info())


@MODELS.register_module()
class ChipAlgorithm(BaseAlgorithm):

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator=dict(
                     type='ChipMutator',
                     channel_unit_cfg=dict(type='ChipUnit')),
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.mutator: ChipMutator = MODELS.build(mutator)
        self.mutator.prepare_from_supernet(self.architecture)

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper) -> Dict[str, torch.Tensor]:
        self.eval()
        self.mutator.start_record_info()
        data = self.data_preprocessor(data, False)
        self._run_forward(data, mode='tensor')  # type: ignore
        self.mutator.end_record_info()

        self.mutator.update_predefine_imp()
        self.mutator.reset_recorded_info()
        return {}

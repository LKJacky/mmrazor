# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Type, Union

from mmrazor.models.mutators.channel_mutator.channel_mutator import \
    ChannelMutator
from mmrazor.registry import MODELS
from .unit import BaseCollectUni


@MODELS.register_module()
class BaseCollectMutator(ChannelMutator):

    def __init__(self,
                 channel_unit_cfg: Union[dict, Type[BaseCollectUni]] = dict(
                     type='BaseCollectUni'),
                 parse_cfg: Dict = dict(
                     type='ChannelAnalyzer',
                     demo_input=(1, 3, 224, 224),
                     tracer_type='FxTracer'),
                 **kwargs) -> None:
        super().__init__(channel_unit_cfg, parse_cfg, **kwargs)

    def start_record_info(self) -> None:
        """Start recording the related information."""
        for unit in self.mutable_units:
            unit.start_record_fisher_info()

    def end_record_info(self) -> None:
        """Stop recording the related information."""
        for unit in self.mutable_units:
            unit.end_record_fisher_info()

    def reset_recorded_info(self) -> None:
        """Reset the related information."""
        for unit in self.mutable_units:
            unit.reset_recorded()

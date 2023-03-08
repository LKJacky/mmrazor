# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from mmengine.model.utils import _BatchNormXd
from mmengine.utils.dl_utils.parrots_wrapper import \
    SyncBatchNorm as EngineSyncBatchNorm

import mmrazor.models.architectures.dynamic_ops as dynamic_ops
from mmrazor.models.mutables.mutable_channel.mutable_channel_container import \
    MutableChannelContainer
from mmrazor.models.mutables.mutable_channel.units.l1_mutable_channel_unit import \
    L1MutableChannelUnit  # noqa
from mmrazor.registry import MODELS
from .ops import CollectConv2d, CollectLinear, CollectMixin


class CollectUnitMixin:

    @property
    def input_related_collect_ops(self):
        for channel in self.input_related:
            if isinstance(channel.module, CollectMixin):
                yield channel.module

    @property
    def output_related_collect_ops(self):
        for channel in self.output_related:
            if isinstance(channel.module, CollectMixin):
                yield channel.module

    @property
    def collect_ops(self):
        for module in self.input_related_collect_ops:
            yield module
        for module in self.output_related_collect_ops:
            yield module

    # fisher information recorded

    def start_record_fisher_info(self) -> None:
        """Start recording the related fisher info of each channel."""
        for module in self.collect_ops:
            module.start_record()

    def end_record_fisher_info(self) -> None:
        """Stop recording the related fisher info of each channel."""
        for module in self.collect_ops:
            module.end_record()

    def reset_recorded(self) -> None:
        """Reset the recorded info of each channel."""
        for module in self.collect_ops:
            module.reset_recorded()


@MODELS.register_module()
class BaseCollectUni(L1MutableChannelUnit, CollectUnitMixin):

    def __init__(self, num_channels: int, *args) -> None:
        super().__init__(num_channels, *args)

    def prepare_for_pruning(self, model: nn.Module) -> None:
        """Prepare for pruning, including register mutable channels.

        Args:
            model (nn.Module): The model need to be pruned.
        """
        # register MutableMask
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: CollectConv2d,
                nn.BatchNorm2d: dynamic_ops.DynamicBatchNorm2d,
                nn.Linear: CollectLinear,
                nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                EngineSyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                _BatchNormXd: dynamic_ops.DynamicBatchNormXd,
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    @property
    def input_related_dynamic_ops(self):
        for channel in self.input_related:
            if isinstance(channel.module, CollectMixin):
                yield channel.module

    @property
    def output_related_dynamic_ops(self):
        for channel in self.output_related:
            if isinstance(channel.module, CollectMixin):
                yield channel.module

    @property
    def dynamic_ops(self):
        for module in self.input_related_dynamic_ops:
            yield module
        for module in self.output_related_dynamic_ops:
            yield module

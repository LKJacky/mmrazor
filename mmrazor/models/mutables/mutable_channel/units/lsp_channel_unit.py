# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmengine.model.utils import _BatchNormXd
from mmengine.utils.dl_utils.parrots_wrapper import \
    SyncBatchNorm as EngineSyncBatchNorm

from mmrazor.models.architectures import dynamic_ops
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from .l1_mutable_channel_unit import L1MutableChannelUnit
from .lsp import quick_back_cssp
from .mutable_channel_unit import Channel


@MODELS.register_module()
class LSPMutableChannelUnit(L1MutableChannelUnit):

    def __init__(self,
                 num_channels: int,
                 choice_mode='number',
                 divisor=1,
                 min_value=1,
                 min_ratio=0.9,
                 use_double=False) -> None:
        super().__init__(num_channels, choice_mode, divisor, min_value,
                         min_ratio)
        self.input_fm = None
        self.hook_handles: List = []
        self.imp = torch.rand([num_channels])
        self.recorded_num = 0
        self.use_double = use_double

    @property
    def current_choice(self):
        return super().current_choice

    @current_choice.setter
    def current_choice(self, value):
        int_choice = self._get_valid_int_choice(value)
        if int_choice == self.mutable_channel.activated_channels:
            return
        mask = self._generate_mask(int_choice)
        self.mutable_channel.current_choice = mask
        assert self.input_fm is not None
        for channel in self.input_related:
            if isinstance(channel.module, dynamic_ops.LSPDynamicConv2d):
                channel.module.refresh_weight(self.input_fm)
            elif isinstance(channel.module, dynamic_ops.LSPDynamicLinear):
                channel.module.refresh_weight(self.input_fm)
            else:
                pass

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: dynamic_ops.LSPDynamicConv2d,
                nn.BatchNorm2d: dynamic_ops.DynamicBatchNorm2d,
                nn.Linear: dynamic_ops.LSPDynamicLinear,
                nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                EngineSyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                _BatchNormXd: dynamic_ops.LSPDynamicConv2d,
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)
        self.start_colloct_inputs()

    def foward_hook_wrap(self, start=0, end=-1):

        def forward_hook(module, inputs, output):

            input: torch.Tensor = inputs[0]
            B = input.shape[0]
            self.recorded_num += B
            if isinstance(module, nn.Conv2d):
                input = input.permute([1, 0, 2, 3]).flatten(1)
            elif isinstance(module, nn.Linear):
                input = input.permute([1, 0]).flatten(1)
            else:
                raise NotImplementedError()

            self.add_input_fm(input[start:end])
            if self.recorded_num > 512:
                self.compute_imp()
                self.end_colloct_inputs()

        return forward_hook

    def add_input_fm(self, tensor):
        with torch.no_grad():
            tensor = tensor.detach()
            assert len(tensor.shape) == 2
            if self.input_fm is None:
                self.input_fm = tensor
            elif isinstance(self.input_fm, torch.Tensor):
                self.input_fm = torch.cat([self.input_fm, tensor], dim=-1)
            else:
                raise NotImplementedError()

            if self.input_fm.shape[0] > 512:
                self.compute_imp()
                self.end_colloct_inputs()

    def start_colloct_inputs(self):
        for channel in self.input_related:
            if isinstance(channel.module, nn.Conv2d) or isinstance(
                    channel.module, nn.Linear):
                handle = channel.module.register_forward_hook(
                    self.foward_hook_wrap(channel.start, channel.end))
                self.hook_handles.append(handle)

    def end_colloct_inputs(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def compute_imp(self):
        assert self.input_fm is not None
        from mmengine import MMLogger
        MMLogger.get_current_instance().info('update lsp imp')
        if self.use_double:
            _, loss = quick_back_cssp(self.input_fm.transpose(-1, -2).double())
            loss = loss.float()
        else:
            _, loss = quick_back_cssp(self.input_fm.transpose(-1, -2))
        self.imp = loss

    def _generate_mask(self, choice: int) -> torch.Tensor:
        assert self.imp is not None
        imp = self.imp
        idx = (imp * -1).topk(choice)[1]
        mask = torch.zeros([self.num_channels]).to(idx.device)
        mask.scatter_(0, idx, 1)
        return mask

    @property
    def is_mutable(self) -> bool:

        def traverse(channels: List[Channel]):
            has_dynamic_op = False
            all_channel_prunable = True
            for channel in channels:
                if channel.is_mutable is False:
                    all_channel_prunable = False
                    break

                if isinstance(channel.module, dynamic_ops.DynamicChannelMixin):
                    has_dynamic_op = True
            return has_dynamic_op, all_channel_prunable

        input_has_dynamic_op, input_all_prunable = traverse(self.input_related)
        output_has_dynamic_op, output_all_prunable = traverse(
            self.output_related)
        for channel in self.input_related:
            if (isinstance(channel.module, nn.Linear)
                    and channel.num_channels != channel.module.in_features):
                input_all_prunable = False
            if (isinstance(channel.module, nn.Conv2d)
                    and channel.num_channels != channel.module.in_channels):
                input_all_prunable = False

        return len(self.output_related) > 0 \
            and len(self.input_related) > 0 \
            and input_has_dynamic_op \
            and input_all_prunable \
            and output_has_dynamic_op \
            and output_all_prunable

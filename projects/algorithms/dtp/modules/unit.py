from typing import Union

import torch
import torch.nn as nn

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.mutables import L1MutableChannelUnit
from mmrazor.registry import MODELS
from .mutable_channels import (DTPMutableChannelImp,
                               ImpMutableChannelContainer, PASMutableChannel,
                               grad_clip_wrapper)
from .ops import ImpBatchnorm2d, ImpConv2d, ImpLinear


@MODELS.register_module()
class ImpUnit(L1MutableChannelUnit):

    def __init__(
        self,
        num_channels: int,
        imp_type='l1',
        grad_clip=-1,
    ) -> None:
        super().__init__(num_channels, choice_mode='number')

        assert imp_type in ['dtp', 'pas']
        self.imp_type = imp_type

        self.mutable_channel: Union[PASMutableChannel, DTPMutableChannelImp]
        if self.imp_type == 'pas':
            self.mutable_channel = PASMutableChannel(self.num_channels)
        else:
            self.mutable_channel = DTPMutableChannelImp(  # noqa
                self.num_channels)

        self.requires_grad_(False)

        self.grad_clip = grad_clip

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: ImpConv2d,
                nn.BatchNorm2d: ImpBatchnorm2d,
                nn.Linear: ImpLinear,
                nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
            })
        self._register_channel_container(model, ImpMutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    @torch.no_grad()
    def resort(self):
        if isinstance(self.mutable_channel, DTPMutableChannelImp):
            norm = self._get_unit_norm()
            imp = norm

            index = imp.sort(descending=True)[1]
            new_index = self.mutable_channel.index.gather(0, index)
            self.mutable_channel.index.data = new_index

    @torch.no_grad()
    def importance(self):
        if self.imp_type == 'dtp':
            return self.mutable_channel.current_imp
        elif self.imp_type == 'pas':
            return self.mutable_channel.imp.detach()
        else:
            raise NotImplementedError()

    def activate_grad(self):
        self.requires_grad_(True)
        if isinstance(self.mutable_channel, DTPMutableChannelImp):
            self.mutable_channel.e.register_hook(
                grad_clip_wrapper(self.grad_clip))

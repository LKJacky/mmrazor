import torch.nn as nn

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.mutables import MutableChannelContainer


class ExpandMixin:

    def expand(self, zero=False):
        pass

    @property
    def _mutable_in_channel(self):
        pass

    @property
    def _mutable_out_channel(self):
        pass

    @property
    def mutable_in_channel(self):
        if self.in_mutable is not None:
            return self.in_mutable.current_mask.numel()
        else:
            return self._mutable_in_channel

    @property
    def mutable_out_channel(self):
        if self.out_mutable is not None:
            return self.out_mutable.current_mask.numel()
        else:
            return self._mutable_out_channel

    @property
    def in_mutable(self) -> MutableChannelContainer:
        return self.get_mutable_attr('in_channels')  # type: ignore

    @property
    def out_mutable(self) -> MutableChannelContainer:
        return self.get_mutable_attr('out_channels')  # type: ignore

    def zero_weight_(self: nn.Module):
        for p in self.parameters():
            p.data.zero_()


class ExpandConv2d(dynamic_ops.DynamicConv2d, ExpandMixin):

    @property
    def _mutable_in_channel(self):
        return self.in_channels

    @property
    def _mutable_out_channel(self):
        return self.out_channels

    def expand(self, zero=False):
        module = nn.Conv2d(self.mutable_in_channel, self.mutable_out_channel,
                           self.kernel_size, self.stride, self.padding,
                           self.dilation, self.groups, self.bias is not None,
                           self.padding_mode)
        if zero:
            self.zero_weight_()
        module.weight.data[:self.out_channels, :self.
                           in_channels] = self.weight  # out,in
        if module.bias is not None:
            module.bias.data[:self.out_channels] = self.bias
        return module


class ExpandLinear(dynamic_ops.DynamicLinear, ExpandMixin):

    @property
    def _mutable_in_channel(self):
        return self.in_features

    @property
    def _mutable_out_channel(self):
        return self.out_features

    def expand(self, zero=False):
        module = nn.Linear(self.mutable_in_channel, self.mutable_out_channel,
                           self.bias is not None)
        if zero:
            self.zero_weight_()
        module.weight.data[:self.out_features, :self.in_features] = self.weight
        if module.bias is not None:
            module.bias.data[:self.out_features] = self.bias
        return module


class ExpandBn2d(dynamic_ops.DynamicBatchNorm2d, ExpandMixin):

    @property
    def _mutable_in_channel(self):
        return self.num_features

    @property
    def _mutable_out_channel(self):
        return self.num_features

    def expand(self, zero=False):
        module = nn.BatchNorm2d(self.mutable_in_channel, self.eps,
                                self.momentum, self.affine,
                                self.track_running_stats)
        if zero:
            self.zero_weight_
        if module.running_mean is not None:
            module.running_mean.data[:self.num_features] = self.running_mean
        if module.running_var is not None:
            module.running_var.data[:self.num_features] = self.running_var
        self.weight.data[:self.num_features] = self.weight
        self.bias.data[:self.num_features] = self.bias
        return module

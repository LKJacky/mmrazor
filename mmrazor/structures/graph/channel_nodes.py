# Copyright (c) OpenMMLab. All rights reserved.

import operator
from abc import abstractmethod
from typing import List, Union

import torch
import torch.nn as nn
from mmengine import MMLogger

from .channel_flow import ChannelTensor
from .module_graph import ModuleNode


class ChannelNode(ModuleNode):
    """A ChannelNode is like a torch module. It accepts  a ChannelTensor and
    output a ChannelTensor. The difference is that the torch module transforms
    a tensor, while the ChannelNode records the information of channel
    dependency in the ChannelTensor.

    Args:
        name (str): The name of the node.
        val (Union[nn.Module, str]): value of the node.
        expand_ratio (int, optional): expand_ratio compare with channel
            mask. Defaults to 1.
        module_name (str, optional): the module name of the module of the
            node.
    """

    # init

    def __init__(self,
                 name: str,
                 val: Union[nn.Module, str],
                 module_name='') -> None:

        super().__init__(name, val, module_name)
        self.in_channel_tensor: Union[None, ChannelTensor] = None
        self.out_channel_tensor: Union[None, ChannelTensor] = None

    @classmethod
    def copy_from(cls, node):
        """Copy from a ModuleNode."""
        assert isinstance(node, ModuleNode)
        return cls(node.name, node.val, node.module_name)

    def reset_channel_tensors(self):
        """Reset the owning ChannelTensors."""
        self.in_channel_tensor = None
        self.out_channel_tensor = None

    # forward

    def forward(self, in_channel_tensors=None):
        """Forward with ChannelTensors."""
        if in_channel_tensors is None:
            out_channel_tensors = [
                node.out_channel_tensor for node in self.prev_nodes
            ]
            in_channel_tensors = out_channel_tensors
        self.channel_forward(in_channel_tensors)

    @abstractmethod
    def channel_forward(self, channel_tensors: List[ChannelTensor]):
        """Forward with ChannelTensors."""
        assert len(channel_tensors) == 1, f'{len(channel_tensors)}'

        self.in_channel_tensor = channel_tensors[0]
        self.out_channel_tensor = ChannelTensor(self.out_channels)

    # channels

    # @abstractmethod
    @property
    def in_channels(self) -> int:
        """Get the number of input channels of the node."""
        try:
            return self._in_channels
        except NotImplementedError:
            return \
                self._get_in_channels_by_prev_nodes(self.prev_nodes)

    # @abstractmethod
    @property
    def out_channels(self) -> int:
        """Get the number of output channels of the node."""
        try:
            return self._out_channels
        except NotImplementedError:
            return self._get_out_channel_by_in_channels(self.in_channels)

    def check_channel(self):
        for node in self.prev_nodes:
            assert node.out_channels == self.in_channels, \
                f'{self} and {node} dismatch'

    @property
    def _in_channels(self) -> int:
        raise NotImplementedError(
            f'{self.name}({self.__class__.__name__}) has no _in_channels')

    @property
    def _out_channels(self) -> int:
        raise NotImplementedError(
            f'{self.name}({self.__class__.__name__}) has no _out_channels')

    def _get_out_channel_by_in_channels(self, in_channels):
        return in_channels

    def _get_in_channels_by_prev_nodes(self, prev_nodes):
        if len(prev_nodes) == 0:
            from mmengine import MMLogger
            MMLogger.get_current_instance().warning(
                (f'As {self.name}'
                 'has no prev nodes, so we set the in channels of it to 3.'))
            return 3
        else:
            return prev_nodes[0].out_channels

    def __repr__(self) -> str:
        return f'{self.name}_({self.in_channels},{self.out_channels})'


# basic nodes


class PassChannelNode(ChannelNode):
    """A PassChannelNode has the same number of input channels and output
    channels.

    Besides, the corresponding input channels and output channels belong to one
    channel unit. Such as  BatchNorm, Relu.
    """

    def channel_forward(self, channel_tensors: List[ChannelTensor]):
        """Channel forward."""
        PassChannelNode._channel_forward(self, channel_tensors[0])

    def __repr__(self) -> str:
        return super().__repr__() + '_pass'

    @staticmethod
    def _channel_forward(node: ChannelNode, tensor: ChannelTensor):
        """Channel forward."""
        assert node.in_channels == node.out_channels
        assert isinstance(tensor, ChannelTensor)
        node.in_channel_tensor = tensor
        node.out_channel_tensor = tensor


class MixChannelNode(ChannelNode):
    """A MixChannelNode  has independent input channels and output channels."""

    def channel_forward(self, channel_tensors: List[ChannelTensor]):
        """Channel forward."""
        assert len(channel_tensors) <= 1
        if len(channel_tensors) == 1:
            self.in_channel_tensor = channel_tensors[0]
            self.out_channel_tensor = ChannelTensor(self.out_channels)
        else:
            raise NotImplementedError()

    def __repr__(self) -> str:
        return super().__repr__() + '_mix'


class BindChannelNode(ChannelNode):
    """A BindChannelNode has multiple inputs, and all input channels belong to
    the same channel unit."""

    def channel_forward(self, channel_tensors: List[ChannelTensor]):
        """Channel forward."""
        assert len(channel_tensors) > 0, f'{self}'
        #  align channel_tensors
        for tensor in channel_tensors[1:]:
            channel_tensors[0].union(tensor)
        self.in_channel_tensor = channel_tensors[0]
        self.out_channel_tensor = channel_tensors[0]

    def __repr__(self) -> str:
        return super().__repr__() + '_bind'


class CatChannelNode(ChannelNode):
    """A CatChannelNode cat all input channels."""

    def channel_forward(self, channel_tensors: List[ChannelTensor]):
        tensor_cat = ChannelTensor.cat(channel_tensors)
        self.in_channel_tensor = tensor_cat
        self.out_channel_tensor = tensor_cat

    def check_channel(self):
        in_num = [node.out_channels for node in self.prev_nodes]
        assert sum(in_num) == self.in_channels, (
            f'the channel of {self} dismatched with prev nodes')

    def _get_in_channels_by_prev_nodes(self, prev_nodes):
        assert len(prev_nodes) > 0
        nums = [node.out_channels for node in prev_nodes]
        return sum(nums)

    def __repr__(self) -> str:
        return super().__repr__() + '_cat'


class ExpandChannelNode(ChannelNode):

    def __init__(self,
                 name: str,
                 val: Union[nn.Module, str],
                 module_name='',
                 expand_ratio=1) -> None:
        super().__init__(name, val, module_name)
        self.expand_ratio = expand_ratio

    def _get_out_channel_by_in_channels(self, in_channels):
        return in_channels * self.expand_ratio

    def channel_forward(self, channel_tensors: List[ChannelTensor]):
        assert len(channel_tensors) == 1, f'{self}'
        assert self.out_channels >= self.in_channels, f'{self}'
        assert self.out_channels % self.in_channels == 0, f'{self}'
        tensor0 = channel_tensors[0]
        self.in_channel_tensor = tensor0
        self.out_channel_tensor = tensor0.expand(self.expand_ratio)


# module nodes


class ConvNode(MixChannelNode):
    """A ConvNode corresponds to a Conv2d module.

    It can deal with normal conv, dwconv and gwconv.
    """

    def __init__(self,
                 name: str,
                 val: Union[nn.Module, str],
                 module_name='') -> None:
        super().__init__(name, val, module_name)
        assert isinstance(self.val, nn.Conv2d)

    @property
    def conv_type(self):
        if self.val.groups == 1:
            return 'conv'
        elif self.val.in_channels == self.out_channels == self.val.groups:
            return 'dwconv'
        else:
            return 'gwconv'

    def channel_forward(self, channel_tensors: List[ChannelTensor]):
        if self.conv_type == 'conv':
            return super().channel_forward(channel_tensors)
        elif self.conv_type == 'dwconv':
            return PassChannelNode._channel_forward(self, channel_tensors[0])
        elif self.conv_type == 'gwconv':
            return self._gw_conv_channel_forward(channel_tensors)
        else:
            raise NotImplementedError(f'{self}')

    def _gw_conv_channel_forward(self, channel_tensors: List[ChannelTensor]):

        def group_union(tensor: ChannelTensor, groups: int):
            c_per_group = len(tensor) // groups
            group_tensor = ChannelTensor(c_per_group)
            for i in range(groups):
                tensor[i * c_per_group:(i + 1) *
                       c_per_group].union(group_tensor)

        assert len(channel_tensors) == 1
        tensor0 = channel_tensors[0]
        conv: nn.Conv2d = self.val
        group_union(tensor0, conv.groups)
        self.in_channel_tensor = tensor0
        self.out_channel_tensor = ChannelTensor(self.out_channels)
        group_union(self.out_channel_tensor, conv.groups)

    @property
    def _in_channels(self) -> int:
        return self.val.in_channels

    @property
    def _out_channels(self) -> int:
        return self.val.out_channels

    def __repr__(self) -> str:
        return super().__repr__() + '_conv'


class LinearNode(MixChannelNode):
    """A LinearNode corresponds to a Linear module."""

    def __init__(self,
                 name: str,
                 val: Union[nn.Module, str],
                 module_name='') -> None:
        super().__init__(name, val, module_name)
        assert isinstance(self.val, nn.Linear)

    @property
    def _in_channels(self) -> int:
        return self.val.in_features

    @property
    def _out_channels(self) -> int:
        return self.val.out_features

    def __repr__(self) -> str:
        return super().__repr__() + 'linear'


class NormNode(PassChannelNode):
    """A NormNode corresponds to a BatchNorm2d module."""

    def __init__(self,
                 name: str,
                 val: Union[nn.Module, str],
                 module_name='') -> None:
        super().__init__(name, val, module_name)
        assert isinstance(self.val, nn.BatchNorm2d)

    @property
    def _in_channels(self) -> int:
        return self.val.num_features

    @property
    def _out_channels(self) -> int:
        return self.val.num_features

    def __repr__(self) -> str:
        return super().__repr__() + '_bn'


# converter


def default_channel_node_converter(node: ModuleNode) -> ChannelNode:
    """The default node converter for ChannelNode."""

    def warn(default='PassChannelNode'):
        logger = MMLogger('mmrazor', 'mmrazor')
        logger.warn((f"{node.name}({node.val}) node can't find match type of"
                     'channel_nodes,'
                     f'replaced with {default} by default.'))

    module_mapping = {
        nn.Conv2d: ConvNode,
        nn.BatchNorm2d: NormNode,
        nn.Linear: LinearNode,
        nn.modules.ReLU: PassChannelNode,
        nn.modules.Hardtanh: PassChannelNode,
        # pools
        nn.modules.pooling._AvgPoolNd: PassChannelNode,
        nn.modules.pooling._AdaptiveAvgPoolNd: PassChannelNode,
        nn.modules.pooling._MaxPoolNd: PassChannelNode,
        nn.modules.pooling._AdaptiveMaxPoolNd: PassChannelNode,
    }
    function_mapping = {
        torch.add: BindChannelNode,
        torch.cat: CatChannelNode,
        operator.add: BindChannelNode
    }
    name_mapping = {
        'bind_placeholder': BindChannelNode,
        'pass_placeholder': PassChannelNode,
        'cat_placeholder': CatChannelNode,
    }
    if isinstance(node.val, nn.Module):
        # module_mapping
        for module_type in module_mapping:
            if isinstance(node.val, module_type):
                return module_mapping[module_type].copy_from(node)

    elif isinstance(node.val, str):
        for module_type in name_mapping:
            if node.val == module_type:
                return name_mapping[module_type].copy_from(node)

    else:
        for fun_type in function_mapping:
            if node.val == fun_type:
                return function_mapping[fun_type].copy_from(node)
    if len(node.prev_nodes) > 1:
        warn('BindChannelNode')
        return BindChannelNode.copy_from(node)
    else:
        warn('PassChannelNode')
        return PassChannelNode.copy_from(node)

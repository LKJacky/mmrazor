# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, Dict, List

from torch.nn import Module

from .base_graph import BaseGraph
from .channel_flow import ChannelTensor
from .channel_nodes import ChannelNode, default_channel_node_converter
from .module_graph import ModuleGraph


class ChannelGraph(ModuleGraph[ChannelNode]):
    """ChannelGraph is used to trace the channel dependency of a model.

    A ChannelGraph generates a ChannelTensor as the input to the model. Then,
    the tensor can forward through all nodes and collect channel dependency.
    """

    @classmethod
    def copy_from(cls,
                  graph: 'BaseGraph',
                  node_converter: Callable = default_channel_node_converter):
        """Copy from a ModuleGraph."""
        assert isinstance(graph, ModuleGraph)
        return super().copy_from(graph, node_converter)

    def collect_units(self) -> Dict:
        """Collect channel units in the graph.
        "hash"{
            'channels':{
                'input_related':[
                    {
                        "name":"backbone.bn1",
                        "start":0,
                        "end":64,
                        "expand_ratio":1,
                        "is_output_channel":false
                    }
                ],
                'output_related':[
                    ...
                ]
            }
        }"""
        chanel_config_template: Dict = {
            'channels': {
                'input_related': [],
                'output_related': []
            }
        }

        def process_tensor(node: ChannelNode, is_output_tensor,
                           unit_hash_dict: Dict):
            if is_output_tensor:
                tensor = node.out_channel_tensor
            else:
                tensor = node.in_channel_tensor
            assert tensor is not None
            for (start, end), hash in tensor.elems_hash_dict.items():
                channel_config = {
                    'name': node.name,
                    'start': start,
                    'end': end,
                    'is_output_channel': is_output_tensor
                }
                if hash not in unit_hash_dict:
                    unit_hash_dict[hash] = copy.deepcopy(
                        chanel_config_template)
                unit_hash_dict[hash][
                    'channels']['output_related' if is_output_tensor else
                                'input_related'].append(channel_config)

        unit_hash_dict: Dict = {}

        for node in self.topo_traverse():
            process_tensor(node, True, unit_hash_dict)
            process_tensor(node, False, unit_hash_dict)

        return unit_hash_dict

    def forward(self, num_input_channel=3):
        """Generate a ChanneelTensor and let it forwards through the graph."""
        for node in self.topo_traverse():
            node.reset_channel_tensors()
        for i, node in enumerate(self.topo_traverse()):
            node: ChannelNode
            if len(node.prev_nodes) == 0:
                tensor = ChannelTensor(num_input_channel)
                node.forward([tensor])
            else:
                node.forward()
        self._merge_same_module()

    def _merge_same_module(self):
        """Union all nodes with the same module to the same unit."""
        module2node: Dict[Module, List[ChannelNode]] = dict()
        for node in self:
            if isinstance(node.val, Module):
                if node.val not in module2node:
                    module2node[node.val] = []
                if node not in module2node[node.val]:
                    module2node[node.val].append(node)

        for module in module2node:
            if len(module2node[module]) > 1:
                nodes = module2node[module]
                assert nodes[0].in_channel_tensor is not None and \
                    nodes[0].out_channel_tensor is not None
                for node in nodes[1:]:
                    nodes[0].in_channel_tensor.union(node.in_channel_tensor)
                    nodes[0].out_channel_tensor.union(node.out_channel_tensor)

import copy

import torch.nn as nn
from mmengine import dist
from mmengine.model.utils import convert_sync_batchnorm

from mmrazor.models import BaseAlgorithm
from mmrazor.models.mutators import ChannelMutator
from mmrazor.registry import MODELS
from ...chex.modules.chex_algorithm import ChexAlgorithm  # type: ignore
from ...chex.modules.chex_mutator import ChexMutator  # type: ignore


def get_structure(model: nn.Module):
    model = copy.deepcopy(model)
    from projects.cores.expandable_ops.unit import ExpandableUnit
    mutator = ChannelMutator[ExpandableUnit](channel_unit_cfg=ExpandableUnit)
    mutator.prepare_from_supernet(model)
    return mutator.choice_template


def expand_static_model(model: nn.Module, structure):
    """Expand the channels of a model.

    Args:
        model (nn.Module): the model to be expanded.
        divisor (_type_): the divisor to make the channels divisible.

    Returns:
        nn.Module: an expanded model.
    """
    from projects.cores.expandable_ops.unit import ExpandableUnit, expand_model
    state_dict = model.state_dict()
    mutator = ChannelMutator[ExpandableUnit](channel_unit_cfg=ExpandableUnit)
    mutator.prepare_from_supernet(model)
    model.load_state_dict(state_dict)
    for unit in mutator.mutable_units:
        if unit.name in structure:
            unit.expand_to(structure[unit.name])
    expand_model(model, zero=True)


@MODELS.register_module()
class ChexRAlgorithm(ChexAlgorithm):

    def __init__(self,
                 architecture,
                 mutator_cfg=dict(
                     type='ChexMutator',
                     channel_unit_cfg=dict(type='ChexUnit')),
                 delta_t=2,
                 total_steps=10,
                 init_growth_rate=0.3,
                 model_expand_ratio=0.0,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(
            architecture,
            mutator_cfg,
            delta_t,
            total_steps,
            init_growth_rate,
            data_preprocessor,
            init_cfg,
        )

        BaseAlgorithm.__init__(self, architecture, data_preprocessor, init_cfg)
        self.expand_model(model_expand_ratio)

        if dist.is_distributed():
            self.architecture = convert_sync_batchnorm(self.architecture)

        self.delta_t = delta_t
        self.total_steps = total_steps
        self.init_growth_rate = init_growth_rate

        self.mutator: ChexMutator = MODELS.build(mutator_cfg)
        self.mutator.prepare_from_supernet(self.architecture)

    def expand_model(self, ratio):
        if ratio == 0.0:
            return
        else:
            choices = get_structure(self.architecture)
            for k, v in choices.items():
                choices[k] = int(v * (1 + ratio))
            expand_static_model(self.architecture, choices)

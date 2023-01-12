# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch.nn as nn

from mmrazor.models.mutables import BaseMutable, SquentialMutableChannel
from .dynamic_utils import MutableDepth
from .searchable_modules import TinyNasBackbone


def smart_round(x, base=8):
    if base is None:
        if x > 32 * 8:
            round_base = 32
        elif x > 16 * 8:
            round_base = 16
        else:
            round_base = 8
    else:
        round_base = base

    return max(round_base, round(x / float(round_base)) * round_base)


class DamoMutator():

    def __init__(self) -> None:
        self.search_space = None
        self.mutables: list = []

    def prepare_for_supernet(self, model: TinyNasBackbone):
        model.init_search_space()
        self.search_space = model.search_space()
        self.mutables = self._get_mutables(self.search_space)

    def sample_subnet(self):
        assert self.search_space is not None
        for mutable in self.mutables:
            if isinstance(mutable, BaseMutable) and mutable.is_fixed is False:
                if isinstance(mutable, SquentialMutableChannel):
                    mutable.current_choice = self.mutate_channel(mutable)
                elif isinstance(mutable, MutableDepth):
                    mutable.current_choice = self.mutable_depth(mutable)

    # mutate mutables

    def mutate_channel(self, mutable: SquentialMutableChannel):
        # search_channel_list = [2.0, 1.5, 1.25, 0.8, 0.6, 0.5]
        # ratio = random.choice(search_channel_list)
        # channel_num = smart_round(mutable.num_channels * ratio)
        channel_num = random.randint(1, mutable.num_channels)
        return channel_num

    def mutable_depth(self, mutable: MutableDepth):
        search_layer_list = [-2, -1, 1, 2]
        new_depth = mutable.current_choice
        for _ in range(len(search_layer_list)):
            bias = random.choice(search_layer_list)
            new_depth = mutable.current_choice + bias
            new_depth = max(1, new_depth)
            if new_depth != mutable.current_choice:
                break
        return new_depth

    # private methods

    def _get_mutables(self, search_space):
        mutables = []

        def tranverse(mutable_dict: nn.ModuleDict):
            for value in mutable_dict.values():
                if isinstance(value, BaseMutable):
                    mutables.append(value)
                elif isinstance(value, nn.ModuleDict):
                    tranverse(value)
                else:
                    raise NotImplementedError()

        tranverse(search_space)
        return mutables

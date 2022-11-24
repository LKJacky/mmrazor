# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from ..mixins import DynamicLinearMixin
from .dynamic_conv import normal_equation


class DynamicLinear(nn.Linear, DynamicLinearMixin):
    """Dynamic Linear OP.

    Note:
        Arguments for ``__init__`` of ``DynamicLinear`` is totally same as
        :obj:`torch.nn.Linear`.

    Attributes:
        mutable_in_features (BaseMutable, optional): Mutable for controlling
            ``in_features``.
        mutable_out_features (BaseMutable, optional): Mutable for controlling
            ``out_features``.
    """
    accepted_mutable_attrs = {'in_features', 'out_features'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @property
    def static_op_factory(self):
        return nn.Linear

    @classmethod
    def convert_from(cls, module):
        """Convert a nn.Linear module to a DynamicLinear.

        Args:
            module (:obj:`torch.nn.Linear`): The original Linear module.
        """
        dynamic_linear = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=True if module.bias is not None else False)
        return dynamic_linear

    def forward(self, input: Tensor) -> Tensor:
        """Forward of dynamic linear OP."""
        weight, bias = self.get_dynamic_params()

        return F.linear(input, weight, bias)


class LSPDynamicLinear(DynamicLinear):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.p_weight = self.weight.detach().clone()

    def refresh_weight(self, in_feature=None):

        weight = self.get_linear_proj(
            in_feature, self.mutable_attrs['in_features'].current_choice)

        self.p_weight.data = weight

    def get_linear_proj(self, in_feature: torch.Tensor,
                        select_mask: torch.Tensor):
        with torch.no_grad():
            fileted_feature = in_feature[select_mask]
            proj = normal_equation(
                fileted_feature.transpose(-1, -2),
                in_feature.transpose(-1, -2),
            )  # in in'
            proj = proj.T
            weight = self.weight
            weight = weight @ proj  #

            return weight

    def get_dynamic_params(self: nn.Linear):
        self.p_weight = self.p_weight.to(self.weight.device)
        if ('in_features' not in self.mutable_attrs
                and 'out_features' not in self.mutable_attrs):
            return self.weight, self.bias

        if 'out_features' in self.mutable_attrs:

            out_mask = self.mutable_attrs['out_features'].current_mask.to(
                self.weight.device)
        else:
            out_mask = torch.ones(self.weight.size(0)).bool().to(
                self.weight.device)

        weight = self.p_weight[out_mask]
        bias = self.bias[out_mask] if self.bias is not None else None

        return weight, bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        res = super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                            strict, missing_keys,
                                            unexpected_keys, error_msgs)
        self.p_weight.data = self.weight.data
        return res

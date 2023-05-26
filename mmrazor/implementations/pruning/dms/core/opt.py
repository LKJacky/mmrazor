# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from transformers.models.opt.modeling_opt import (OPTAttention,
                                                  OPTLearnedPositionalEmbedding
                                                  )

from mmrazor.models.architectures.dynamic_ops import (DynamicChannelMixin,
                                                      DynamicLinear)
from mmrazor.models.mutables import BaseMutable


class DynamicEmbedding(nn.Embedding, DynamicChannelMixin):
    attr_mappings: Dict[str, str] = {
        'in_channels': 'num_embeddings',
        'out_channels': 'embedding_dim',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def register_mutable_attr(self, attr: str, mutable: BaseMutable):
        attr = self.attr_mappings[attr]
        self.mutable_attrs[attr] = mutable

    @classmethod
    def convert_from(cls, module: nn.Embedding):
        new_module = cls(module.num_embeddings, module.embedding_dim,
                         module.padding_idx, module.max_norm,
                         module.scale_grad_by_freq, module.sparse)
        new_module.load_state_dict(module.state_dict(), strict=False)
        return new_module

    @property
    def static_op_factory(self):
        return nn.Embedding

    def to_static_op(self) -> Module:
        raise NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight
        if 'embedding_dim' in self.mutable_attrs:
            mutable = self.mutable_attrs['embedding_dim']
            mask = mutable.current_mask
            weight = weight[:, mask]
        return F.embedding(input, weight, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq,
                           self.sparse)


class DynamicOPTLearnedPositionalEmbedding(OPTLearnedPositionalEmbedding,
                                           DynamicChannelMixin):
    attr_mappings: Dict[str, str] = DynamicEmbedding.attr_mappings

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def register_mutable_attr(self, attr: str, mutable: BaseMutable):
        DynamicEmbedding.register_mutable_attr(self, attr, mutable)

    @classmethod
    def convert_from(cls, module: OPTLearnedPositionalEmbedding):
        new_module = cls(module.num_embeddings - module.offset,
                         module.embedding_dim)
        new_module.load_state_dict(module.state_dict(), strict=False)
        return new_module

    @property
    def static_op_factory(self):
        return OPTLearnedPositionalEmbedding

    def to_static_op(self) -> Module:
        raise NotImplementedError()

    def forward(self,
                attention_mask: torch.LongTensor,
                past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (
            torch.cumsum(attention_mask, dim=1).type_as(attention_mask) *
            attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        input = positions + self.offset
        # return super().forward(positions + self.offset)

        return DynamicEmbedding.forward(self, input)


class DynamicOPTAttention(OPTAttention, DynamicChannelMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

        self.q_proj = DynamicLinear.convert_from(self.q_proj)
        self.k_proj = DynamicLinear.convert_from(self.k_proj)
        self.v_proj = DynamicLinear.convert_from(self.v_proj)

        self.out_proj = DynamicLinear.convert_from(self.out_proj)

        self.in_channels = self.q_proj.in_features
        self.out_channels = self.out_proj.out_features

    def register_mutable_attr(self, attr: str, mutable):
        if attr == 'in_channels':
            self.q_proj.register_mutable_attr(attr, mutable)
            self.k_proj.register_mutable_attr(attr, mutable)
            self.v_proj.register_mutable_attr(attr, mutable)
        elif attr == 'out_channels':
            self.out_proj.register_mutable_attr(attr, mutable)
        else:
            raise NotImplementedError()
        self.mutable_attrs[attr] = mutable

    @classmethod
    def convert_from(cls, module: OPTAttention):
        new_module = cls(module.embed_dim, module.num_heads, module.dropout,
                         module.is_decoder, module.q_proj.bias is not None)
        new_module.load_state_dict(module.state_dict(), strict=False)
        return new_module

    def to_static_op(self) -> Module:
        module: OPTAttention = self.static_op_factory(*self.init_args,
                                                      **self.init_kwargs)
        module.k_proj = self.k_proj.to_static_op()
        module.q_proj = self.q_proj.to_static_op()
        module.v_proj = self.v_proj.to_static_op()
        module.out_proj = self.out_proj.to_static_op()
        return module

    @property
    def static_op_factory(self):
        return OPTAttention

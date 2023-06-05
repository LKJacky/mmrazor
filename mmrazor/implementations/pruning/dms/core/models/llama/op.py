# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn as nn
from torch.nn import Module
from transformers.models.llama.modeling_llama import LlamaAttention

from mmrazor.implementations.pruning.dms.core.mutable import (
    ImpMutableChannelContainer, MutableChannelForHead, MutableChannelWithHead,
    MutableHead)
from mmrazor.models.architectures.dynamic_ops import (DynamicChannelMixin,
                                                      DynamicLinear)
from mmrazor.models.mutables import BaseMutable


class DynamicLlamaAttention(LlamaAttention, DynamicChannelMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

        self.q_proj = DynamicLinear.convert_from(self.q_proj)
        self.k_proj = DynamicLinear.convert_from(self.k_proj)
        self.v_proj = DynamicLinear.convert_from(self.v_proj)

        self.o_proj = DynamicLinear.convert_from(self.o_proj)

        self.in_channels = self.q_proj.in_features
        self.out_channels = self.o_proj.out_features

    def register_mutable_attr(self, attr: str, mutable):
        if attr == 'in_channels':
            self.q_proj.register_mutable_attr(attr, mutable)
            self.k_proj.register_mutable_attr(attr, mutable)
            self.v_proj.register_mutable_attr(attr, mutable)
        elif attr == 'out_channels':
            self.o_proj.register_mutable_attr(attr, mutable)
        else:
            raise NotImplementedError()
        self.mutable_attrs[attr] = mutable

    @classmethod
    def convert_from(cls, module: LlamaAttention):
        new_module = cls(module.config)
        new_module.load_state_dict(module.state_dict(), strict=False)
        return new_module

    def to_static_op(self) -> Module:
        module: LlamaAttention = self.static_op_factory(
            *self.init_args, **self.init_kwargs)
        module.k_proj = self.k_proj.to_static_op()
        module.q_proj = self.q_proj.to_static_op()
        module.v_proj = self.v_proj.to_static_op()
        module.o_proj = self.o_proj.to_static_op()
        return module

    @property
    def static_op_factory(self):
        return LlamaAttention

    def init_mutables(self):
        m_head = MutableHead(self.num_heads)
        m_qk = MutableChannelForHead(self.q_proj.out_features, self.num_heads)
        m_v = MutableChannelForHead(self.v_proj.out_features, self.num_heads)
        mutable_qk = MutableChannelWithHead(m_head, m_qk)
        mutable_v = MutableChannelWithHead(m_head, m_v)

        try:
            self.q_proj.register_mutable_attr(
                'out_channels',
                ImpMutableChannelContainer(self.q.out_features))
            self.k_proj.register_mutable_attr(
                'out_channels',
                ImpMutableChannelContainer(self.k.out_features))
            self.v_proj.register_mutable_attr(
                'out_channels',
                ImpMutableChannelContainer(self.v.out_features))
            self.o_proj.register_mutable_attr(
                'in_channels',
                ImpMutableChannelContainer(self.proj.in_features))
        except Exception:
            pass
        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.q_proj, mutable=mutable_qk, is_to_output_channel=True)
        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.k_proj, mutable=mutable_qk, is_to_output_channel=True)
        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.v_proj, mutable=mutable_v, is_to_output_channel=True)

        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.o_proj, mutable_v, is_to_output_channel=False)

        self.attn_mutables = {'head': m_head, 'qk': m_qk, 'v': m_v}

        return m_head, m_qk, m_v

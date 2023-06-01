# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import (OPTAttention,
                                                  OPTDecoderLayer,
                                                  OPTLearnedPositionalEmbedding
                                                  )

from mmrazor.implementations.pruning.dms.core.mutable import (
    ImpMutableChannelContainer, MutableChannelForHead, MutableChannelWithHead,
    MutableHead)
from mmrazor.models.architectures.dynamic_ops import (DynamicChannelMixin,
                                                      DynamicLinear)
from mmrazor.models.mutables import BaseMutable
from ...op import DynamicBlockMixin, ImpLinear, ImpModuleMixin  # type: ignore


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
        out_channel = self.mutable_attrs['embedding_dim'].current_mask
        new_module = nn.Embedding(self.num_embeddings,
                                  out_channel.sum().item(), self.padding_idx,
                                  self.max_norm, self.scale_grad_by_freq,
                                  self.sparse)
        new_module.weight.data = self.weight[:, out_channel]
        return new_module

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
        out_channel = self.mutable_attrs['embedding_dim'].current_mask
        new_module = OPTLearnedPositionalEmbedding(
            self.num_embeddings - self.offset,
            out_channel.sum().item())
        new_module.weight.data = self.weight[:, out_channel]
        return new_module

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


class ImpEmbedding(DynamicEmbedding, ImpModuleMixin):

    def forward(self, input: Tensor) -> Tensor:
        return nn.Embedding.forward(self, input)


class ImpOPTLearnedPositionalEmbedding(DynamicOPTLearnedPositionalEmbedding,
                                       ImpModuleMixin):

    def forward(self,
                attention_mask: torch.LongTensor,
                past_key_values_length: int = 0):
        return OPTLearnedPositionalEmbedding.forward(self, attention_mask,
                                                     past_key_values_length)


class ImpOPTAttention(DynamicOPTAttention, ImpModuleMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

        self.q_proj = ImpLinear.convert_from(self.q_proj)
        self.k_proj = ImpLinear.convert_from(self.k_proj)
        self.v_proj = ImpLinear.convert_from(self.v_proj)

        self.out_proj = ImpLinear.convert_from(self.out_proj)

        self.in_channels = self.q_proj.in_features
        self.out_channels = self.out_proj.out_features

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
            self.out_proj.register_mutable_attr(
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
            self.out_proj, mutable_v, is_to_output_channel=False)

        self.attn_mutables = {'head': m_head, 'qk': m_qk, 'v': m_v}

        return m_head, m_qk, m_v

    def to_static_op(self) -> Module:
        num_heads = int(self.attn_mutables['head'].mask.sum().item())

        module: OPTAttention = OPTAttention(
            embed_dim=self.head_dim * num_heads,
            num_heads=num_heads,
            dropout=self.dropout,
            is_decoder=self.is_decoder,
            bias=self.q_proj.bias is not None)
        module.k_proj = self.k_proj.to_static_op()
        module.q_proj = self.q_proj.to_static_op()
        module.v_proj = self.v_proj.to_static_op()
        module.out_proj = self.out_proj.to_static_op()
        return module


######################################################################


class DynamicOPTDecoderLayer(OPTDecoderLayer, DynamicBlockMixin):

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.config = config
        self._dynamic_block_init()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states * self.scale

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual +
                         hidden_states * self.scale).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs

    def to_static_op(self) -> Module:
        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
        module = OPTDecoderLayer(self.config)
        for name, m in self.named_children():
            assert hasattr(module, name)
            setattr(module, name, _dynamic_to_static(m))
        return module

    @classmethod
    def convert_from(cls, module: OPTDecoderLayer):
        from transformers.activations import ACT2FN
        FN2ACT = {}
        for k, v in ACT2FN.items():
            try:
                hash(v)
                FN2ACT[v] = k
            except Exception:
                pass

        config = OPTConfig(
            hidden_size=module.embed_dim,
            num_attention_heads=module.self_attn.num_heads,
            attention_dropout=module.self_attn.dropout,
            enable_bias=module.fc1.bias is not None,
            do_layer_norm_before=module.do_layer_norm_before,
            dropout=module.dropout,
            activation_function=FN2ACT[type(module.activation_fn)],
            ffn_dim=module.fc1.out_features,
            layer_norm_elementwise_affine=module.final_layer_norm.
            elementwise_affine,
        )
        new_module = cls(config)
        new_module.load_state_dict(module.state_dict())
        return new_module

    @property
    def out_channel(self):
        return self.fc2.out_features

    def quick_flop_forward_hook_wrapper(self):
        """Wrap the hook used in forward."""

        def forward_hook(module, input, output):
            module.quick_flop_recorded_out_shape.append([])
            module.quick_flop_recorded_in_shape.append([])

        return forward_hook


class DynamicOptLayers(nn.Sequential):

    @classmethod
    def convert_from(cls, module: nn.ModuleList):
        new_module = cls()
        for name, module in module.named_children():
            new_module.add_module(name, module)
        return new_module

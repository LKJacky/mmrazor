# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (LlamaConfig,
                                                      LlamaDecoderLayer,
                                                      LlamaModel)

from mmrazor.models.mutables import SequentialMutableChannelUnit
from mmrazor.models.mutables.mutable_channel.units.channel_unit import Channel
from mmrazor.models.mutators import ChannelMutator
from mmrazor.models.task_modules.demo_inputs import BaseDemoInput
from mmrazor.registry import TASK_UTILS


def get_default_in_index(module: nn.Module):
    if isinstance(module, nn.Linear):
        return (0, module.in_features)
    if isinstance(module, nn.LayerNorm):
        return (0, module.normalized_shape[-1])
    if isinstance(module, nn.Embedding):
        return (0, module.num_embeddings)
    else:
        raise NotImplementedError()


def get_default_out_index(module: nn.Module):
    if isinstance(module, nn.Linear):
        return (0, module.out_features)
    if isinstance(module, nn.LayerNorm):
        return (0, module.normalized_shape[-1])
    if isinstance(module, nn.Embedding):
        return (0, module.embedding_dim)
    else:
        raise NotImplementedError(f'{type(module)}')


class OutChannel(Channel):

    def __init__(self, name, module, index=None, node=None) -> None:
        if index is None:
            index = get_default_out_index(module)
        super().__init__(name, module, index, node, is_output_channel=True)


class InChannel(Channel):

    def __init__(self, name, module, index=None, node=None) -> None:
        if index is None:
            index = get_default_in_index(module)
        super().__init__(name, module, index, node, is_output_channel=False)


class LLamaChannelAnalyer():

    def __init__(self, model: LlamaModel) -> None:
        self.model = model

    def get_config(self):

        def parse_layer(module: LlamaDecoderLayer, prefix=''):
            unit1 = SequentialMutableChannelUnit(
                module.mlp.gate_proj.out_features)
            unit1.add_output_related(
                OutChannel(f'{prefix}mlp.gate_proj', module.mlp.gate_proj))
            unit1.add_output_related(
                OutChannel(f'{prefix}mlp.up_proj', module.mlp.up_proj))
            unit1.add_input_related(
                InChannel(f'{prefix}mlp.down_proj', module.mlp.down_proj))

            return {
                f'{prefix}unit1':
                unit1.config_template(with_init_args=True, with_channels=True),
            }

        def parse_model(module: LlamaModel):
            units = {}
            for name, layer in module.layers.named_children():
                units.update(parse_layer(layer, prefix=f'layers.{name}.'))

            return units

        def post_process(config: dict):
            for unit_name in config:
                for key in copy.copy(config[unit_name]['init_args']):
                    if key != 'num_channels':
                        config[unit_name]['init_args'].pop(key)
                config[unit_name].pop('choice')
            return config

        return post_process(parse_model(self.model))


@TASK_UTILS.register_module()
class LlamaDemoInput(BaseDemoInput):

    def _get_data(self, model, input_shape, training):
        data = dict(
            input_ids=torch.randint(0, 32000, input_shape).long(),
            attention_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False)
        return data


if __name__ == '__main__':
    config = LlamaConfig(
        hidden_size=512, intermediate_size=512, num_hidden_layers=5)
    model = LlamaModel(config)
    analyser = LLamaChannelAnalyer(model)
    config = analyser.get_config()
    import json

    print(json.dumps(config, indent=4))
    mutator: ChannelMutator = ChannelMutator(
        parse_cfg={'type': 'Config'},
        channel_unit_cfg={
            'units': config,
            'type': 'SequentialMutableChannelUnit'
        })
    mutator.prepare_from_supernet(model)
    print(len(mutator.mutable_units))
    mutator.set_choices(mutator.sample_choices())
    print(model)

    x = torch.rand([1, 128]).long()
    y = model(x)
    print(y['last_hidden_state'].shape)

    from mmrazor.implementations.pruning.dms.core.mutator import DMSMutator

    config = LlamaConfig(
        hidden_size=512, intermediate_size=512, num_hidden_layers=5)
    model = LlamaModel(config)
    mutator: DMSMutator = DMSMutator(
        prune_qkv=False,
        dtp_mutator_cfg=dict(
            type='DTPAMutator',
            channel_unit_cfg=dict(type='DTPTUnit', default_args=dict()),
            parse_cfg=dict(
                _scope_='mmrazor',
                type='ChannelAnalyzer',
                demo_input=dict(
                    type='LlamaDemoInput',
                    input_shape=(1, 128),
                ),
                tracer_type='FxTracer'),
        ),
    )
    mutator.prepare_from_supernet(model)
    mutator.init_quick_flop(model)
    print(mutator.info())

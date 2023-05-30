# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from transformers import OPTForCausalLM, OPTModel
from transformers.models.opt.modeling_opt import OPTDecoder, OPTDecoderLayer

from mmrazor.models.mutables import SequentialMutableChannelUnit
from mmrazor.models.mutables.mutable_channel.units.channel_unit import Channel
from mmrazor.models.mutators import ChannelMutator


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
        raise NotImplementedError()


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


class OPTChannelAnalyer():

    def __init__(self, model: OPTModel) -> None:
        self.model = model

    def get_config(self):

        def parse_layer(module: OPTDecoderLayer, preffix=''):
            units = {}

            unit_fc = SequentialMutableChannelUnit(module.fc1.out_features)
            unit_fc.add_output_related(OutChannel(preffix + 'fc1', module.fc1))
            unit_fc.add_input_related(InChannel(preffix + 'fc2', module.fc2))
            units[preffix + 'fc'] = unit_fc.config_template(
                with_init_args=True, with_channels=True)

            return units

        def parse_decoder(module: OPTDecoder, preffix=''):
            units = {}
            for name, layer in module.layers.named_children():
                config = parse_layer(
                    layer, preffix=preffix + 'layers.' + name + '.')
                units.update(config)

            return units

        def parse_res_structure(module: OPTDecoder, preffix=''):
            preffix_layer = preffix + 'layers.'
            unit = SequentialMutableChannelUnit(
                module.embed_tokens.embedding_dim)
            unit.add_output_related(
                OutChannel(preffix + 'embed_tokens', module.embed_tokens))
            unit.add_output_related(
                OutChannel(preffix + 'embed_positions',
                           module.embed_positions))
            unit.add_output_related(
                OutChannel(preffix + 'final_layer_norm',
                           module.final_layer_norm))
            for name, layer in module.layers.named_children():
                layer: OPTDecoderLayer  # type: ignore
                unit.add_output_related(
                    OutChannel(preffix_layer + f'{name}.self_attn.out_proj',
                               layer.self_attn.out_proj))
                unit.add_output_related(
                    OutChannel(preffix_layer + f'{name}.fc2', layer.fc2))
                unit.add_output_related(
                    OutChannel(preffix_layer + f'{name}.final_layer_norm',
                               layer.final_layer_norm))

                unit.add_input_related(
                    InChannel(preffix_layer + f'{name}.self_attn.k_proj',
                              layer.self_attn.k_proj))
                unit.add_input_related(
                    InChannel(preffix_layer + f'{name}.self_attn.v_proj',
                              layer.self_attn.v_proj))
                unit.add_input_related(
                    InChannel(preffix_layer + f'{name}.self_attn.q_proj',
                              layer.self_attn.q_proj))
                unit.add_input_related(
                    InChannel(preffix_layer + f'{name}.self_attn_layer_norm',
                              layer.self_attn_layer_norm))
                unit.add_input_related(
                    InChannel(preffix_layer + f'{name}.fc1', layer.fc1))
            return unit.config_template(
                with_channels=True, with_init_args=True)

        def parse_model(module: OPTModel):
            units = {}
            decoder_unit = parse_decoder(module.decoder, preffix='decoder.')
            res_unit = parse_res_structure(module.decoder, preffix='decoder.')

            units['res'] = res_unit
            units.update(decoder_unit)
            return units

        def post_process(config: dict):
            for unit_name in config:
                for key in copy.copy(config[unit_name]['init_args']):
                    if key != 'num_channels':
                        config[unit_name]['init_args'].pop(key)
            return config

        return post_process(parse_model(self.model))


if __name__ == '__main__':
    model: OPTModel = OPTForCausalLM.from_pretrained('facebook/opt-125m').model

    analyser = OPTChannelAnalyer(model)
    config = analyser.get_config()
    import json

    print(json.dumps(config, indent=4))
    # test config

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

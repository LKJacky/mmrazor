# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
from typing import Dict

from mmengine import Config

from mmrazor.models.mutators import ChannelMutator
from mmrazor.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('-o', type=str, default='./prune.py')
    args = parser.parse_args()
    return args


def wrap_prune_config(config: Config, prune_target: Dict,
                      checkpoint_path: str):
    config = copy.deepcopy(config)
    if 'data_preprocessor' in config:
        data_preprocessor = config['data_preprocessor']
    else:
        data_preprocessor = None
    arch_config: Dict = config['model']
    arch_config.update({
        'init_cfg': {
            'type': 'Pretrained',
            'checkpoint': checkpoint_path  # noqa
        },
        'data_preprocessor': data_preprocessor
    })

    algorithm_config = dict(
        _scope_='mmrazor',
        type='ItePruneAlgorithm',
        architecture=arch_config,
        target_pruning_ratio=prune_target,
        mutator_cfg=dict(
            type='ChannelMutator',
            channel_unit_cfg=dict(
                type='L1MutableChannelUnit',
                default_args=dict(choice_mode='ratio')),
            parse_cfg=dict(
                type='BackwardTracer',
                loss_calculator=dict(
                    type='ImageClassifierPseudoLoss',
                    input_shape=(2, 3, 32, 32)))))
    config['model'] = algorithm_config
    config['data_preprocessor'] = None

    return config


if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    checkpoint_path = args.checkpoint
    target_path = args.o

    origin_config = Config.fromfile(config_path)

    # get subnet config
    model = MODELS.build(origin_config['model'])
    mutator: ChannelMutator = ChannelMutator(
        channel_unit_cfg=dict(
            type='L1MutableChannelUnit',
            default_args=dict(choice_mode='ratio')))
    mutator.prepare_from_supernet(model)
    choice_template = mutator.choice_template

    # prune and finetune

    prune_config: Config = wrap_prune_config(origin_config, choice_template,
                                             checkpoint_path)
    prune_config.dump(target_path)

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
from typing import Dict, Tuple

from mmengine import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--flops-min', type=float, default=0.45)
    parser.add_argument('--flops-max', type=float, default=0.55)
    parser.add_argument('-o', type=str, default='./search.py')
    args = parser.parse_args()
    return args


def wrap_search_config(config: Config, checkpoint_path: str,
                       flop_range: Tuple):
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
    model_config = dict(
        _scope_='mmrazor',
        type='SearchWrapper',
        architecture=arch_config,
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

    config['model'] = model_config
    config['data_preprocessor'] = None

    val_loader_config = config['val_dataloader']
    val_loader_config['dataset']['type'] = config[
        'default_scope'] + '.' + val_loader_config['dataset']['type']
    val_evaluator_config = config['val_evaluator']
    val_evaluator_config[
        'type'] = config['default_scope'] + '.' + val_evaluator_config['type']

    searcher_config = dict(
        type='mmrazor.PruneEvolutionSearchLoop',
        dataloader=val_loader_config,
        evaluator=val_evaluator_config,
        max_epochs=20,
        num_candidates=20,
        top_k=5,
        num_mutation=10,
        num_crossover=10,
        mutate_prob=0.2,
        flops_range=flop_range,
        score_key='accuracy/top1')
    config['train_cfg'] = searcher_config
    return config


if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    checkpoint_path = args.checkpoint
    flops_range = (args.flops_min, args.flops_max)
    assert flops_range[1] > flops_range[0]
    target_path = args.o

    origin_config = Config.fromfile(config_path)

    # wrap config for search

    search_config = wrap_search_config(origin_config, checkpoint_path,
                                       flops_range)
    search_config.dump(target_path)

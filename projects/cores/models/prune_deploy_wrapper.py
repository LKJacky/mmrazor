import json
import types

import torch.nn as nn
from mmengine.model import BaseModel, BaseModule

from mmrazor.models import BaseAlgorithm
from mmrazor.models.mutators import ChannelMutator
from mmrazor.registry import MODELS
from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                  load_fix_subnet)
from mmrazor.utils import print_log
from ..expandable_modules.unit import ExpandUnit


def clean_params_init_info(model: nn.Module):
    if hasattr(model, '_params_init_info'):
        delattr(model, '_params_init_info')
    for module in model.modules():
        if hasattr(module, '_params_init_info'):
            delattr(module, '_params_init_info')


def clean_init_cfg(model: BaseModule):
    for module in model.modules():
        if module is model:
            continue
        if isinstance(module, BaseModule):
            module.init_cfg = {}


def empty_init_weights(model):
    pass


def to_static_model(algorithm: BaseAlgorithm):
    if hasattr(algorithm, 'to_static'):
        model = algorithm.to_static()
    else:
        mutables = export_fix_subnet(algorithm.architecture)[0]
        load_fix_subnet(algorithm.architecture, mutables)
        model = algorithm.architecture

    model.data_preprocessor = algorithm.data_preprocessor
    if isinstance(model, BaseModel):
        model.init_cfg = None
        model.init_weights = types.MethodType(empty_init_weights, model)
    return model


@MODELS.register_module()
def PruneDeployWrapper(algorithm, data_preprocessor=None):
    algorithm: BaseAlgorithm = MODELS.build(algorithm)
    algorithm.init_weights()
    clean_params_init_info(algorithm)
    print_log(json.dumps(algorithm.mutator.choice_template, indent=4))

    if hasattr(algorithm, 'to_static'):
        model = algorithm.to_static()
    else:
        mutables = export_fix_subnet(algorithm.architecture)[0]
        load_fix_subnet(algorithm.architecture, mutables)
        model = algorithm.architecture

    model.data_preprocessor = algorithm.data_preprocessor
    if isinstance(model, BaseModel):
        model.init_cfg = None
        model.init_weights = types.MethodType(empty_init_weights, model)
    return model


FinetuneWrapper = PruneDeployWrapper
MODELS.register_module('FinetuneWrapper', module=FinetuneWrapper)


@MODELS.register_module()
def PruneDeployWrapper2(architecture,
                        mutable_cfg={},
                        divisor=1,
                        data_preprocessor=None,
                        init_cfg=None):
    if isinstance(architecture, dict):
        architecture = MODELS.build(architecture)
    assert isinstance(architecture, nn.Module)

    # to dynamic model
    mutator = ChannelMutator[ExpandUnit](channel_unit_cfg=ExpandUnit)
    mutator.prepare_from_supernet(architecture)
    for unit in mutator.mutable_units:
        if unit.name in mutable_cfg:
            unit.current_choice = mutable_cfg[unit.name]
    print_log(json.dumps(mutator.choice_template, indent=4))

    mutables = export_fix_subnet(architecture)[0]
    load_fix_subnet(architecture, mutables)

    if divisor != 1:
        setattr(architecture, '_razor_divisor', divisor)

    return architecture

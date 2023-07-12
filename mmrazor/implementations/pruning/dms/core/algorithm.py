# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmrazor.implementations.pruning.dms.core.models.opt.opt_wrapper import \
    to_static_model
from mmrazor.models import BaseAlgorithm
from mmrazor.registry import MODELS, TASK_UTILS
from mmrazor.utils import RuntimeInfo
from .mutator import DMSMutator
from .scheduler import DMSScheduler
import copy
from mmrazor.models.utils.expandable_utils import expand_expandable_dynamic_model


def convert_float_to_tenosr(res: dict, device):
    for k in res:
        if not isinstance(res[k], torch.Tensor):
            res[k] = torch.tensor(res[k], device=device)
    return res


def update_dict_reverse(config1: dict, config2: dict):
    for key in config2:
        if key in config1 and isinstance(config2[key], dict) and isinstance(
                config1[key], dict):
            update_dict_reverse(config1[key], config2[key])
        else:
            config1[key] = config2[key]
    return config1


class DmsAlgorithmMixin():
    default_mutator_kwargs = dict(
        prune_qkv=False,
        prune_block=False,
        dtp_mutator_cfg=dict(
            type='DTPAMutator',
            channel_unit_cfg=dict(
                type='DTPTUnit', default_args=dict(extra_mapping={})),
            parse_cfg=dict(
                _scope_='mmrazor',
                type='ChannelAnalyzer',
                demo_input=dict(
                    type='DefaultDemoInput',
                    input_shape=(1, 3, 224, 224),
                ),
                tracer_type='FxTracer'),
        ))
    default_scheduler_kargs = dict(
        flops_target=0.8,
        decay_ratio=0.8,
        refine_ratio=0.2,
        flop_loss_weight=1000,
        structure_log_interval=10,
        by_epoch=False,
        target_scheduler='cos',
    )
    expand_unit_config = dict(
        type='ExpandableUnit', default_args=dict(extra_mapping={})),

    def __init__(self,
                 model: nn.Module,
                 mutator_kwargs={},
                 scheduler_kargs={}) -> None:
        mutator_kwargs = update_dict_reverse(
            copy.deepcopy(self.default_mutator_kwargs), mutator_kwargs)
        scheduler_kargs = update_dict_reverse(
            copy.deepcopy(self.default_scheduler_kargs), scheduler_kargs)

        origin_model = copy.deepcopy(model)
        self.architecture = model
        self.mutator: DMSMutator = DMSMutator(**mutator_kwargs)

        self.scheduler = DMSScheduler(
            self.architecture,
            self.mutator,
            **scheduler_kargs,
        )
        self.mutator.channel_depth_train()
        self.architecture.load_state_dict(
            origin_model.state_dict(), strict=False)

        self.runtime_info = None

        self.extra_out = None

    def to_static_model(self):
        self.model = self.architecture
        return to_static_model(self)

    @classmethod
    def expand_model(cls, model: nn.Module, ratio: float):
        mutator_kargs = copy.deepcopy(cls.default_mutator_kwargs)
        mutator_kargs['dtp_mutator_cfg'][
            'channel_unit_cfg'] = cls.expand_unit_config
        mutator = DMSMutator(**mutator_kargs)
        mutator.prepare_from_supernet(model)
        structure = mutator.dtp_mutator.choice_template
        for key, num in structure.items():
            unit = mutator.dtp_mutator._name2unit[key]
            unit.expand_to(int(num * ratio))
        model = expand_expandable_dynamic_model(model, zero=False)
        return model

    @classmethod
    def show_structure(cls, model: nn.Module):
        model = copy.deepcopy(model)
        mutator = DMSMutator(**cls.default_mutator_kwargs)
        mutator.prepare_from_supernet(model)
        mutator.init_quick_flop(model)
        print(mutator.info())


@MODELS.register_module()
class BaseDTPAlgorithm(BaseAlgorithm, DmsAlgorithmMixin):

    def __init__(
            self,
            architecture: Union[BaseModel, Dict],
            mutator_cfg=dict(
                type='DMSMutator',
                channel_unit_cfg=dict(type='ImpUnit', default_args=dict()),
                parse_cfg=dict(
                    _scope_='mmrazor',
                    type='ChannelAnalyzer',
                    demo_input=dict(
                        type='DefaultDemoInput',
                        input_shape=[1, 3, 224, 224],
                    ),
                    tracer_type='FxTracer'),
            ),
            scheduler=dict(
                type='',
                flops_target=0.5,
                decay_ratio=0.6,
                refine_ratio=0.2,
                flop_loss_weight=1,
            ),
            #
            data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
            init_cfg: Optional[Dict] = None) -> None:
        BaseAlgorithm.__init__(self, architecture, data_preprocessor, init_cfg)
        DmsAlgorithmMixin.__init__(
            self,
            self.architecture,
            mutator_kwargs=mutator_cfg,
            scheduler_kargs=scheduler)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor'):
        if self.training and mode == 'loss':
            self.scheduler.before_train_forward(RuntimeInfo.iter(),
                                                RuntimeInfo.epoch(),
                                                RuntimeInfo().max_iters(),
                                                RuntimeInfo().max_epochs())
        res: dict = super().forward(inputs, data_samples, mode)  # type: ignore
        if self.training and mode == 'loss':
            extra_dict = self.scheduler.after_train_forward(
                RuntimeInfo.iter(), RuntimeInfo.epoch(),
                RuntimeInfo().max_iters(),
                RuntimeInfo().max_epochs())
            extra_dict = convert_float_to_tenosr(extra_dict, inputs.device)
            res.update(extra_dict)
        return res


class DmsGeneralAlgorithm(nn.Module, DmsAlgorithmMixin):

    def __init__(
        self,
        model: nn.Module,
        mutator_kwargs=dict(
            prune_qkv=False,
            prune_block=False,
            dtp_mutator_cfg=dict(
                type='DTPAMutator',
                channel_unit_cfg=dict(
                    type='DTPTUnit', default_args=dict(extra_mapping={})),
                parse_cfg=dict(
                    _scope_='mmrazor',
                    type='ChannelAnalyzer',
                    demo_input=dict(
                        type='DefaultDemoInput',
                        input_shape=(1, 3, 224, 224),
                    ),
                    tracer_type='FxTracer'),
            ),
            extra_module_mapping={}),
        scheduler_kargs=dict(
            flops_target=0.8,
            decay_ratio=0.8,
            refine_ratio=0.2,
            flop_loss_weight=1000,
            structure_log_interval=10,
            by_epoch=False,
            target_scheduler='cos',
        )
    ) -> None:
        nn.Module.__init__(self)
        DmsAlgorithmMixin.__init__(self, model, mutator_kwargs,
                                   scheduler_kargs)

    def forward(self, x):
        if self.training:
            self.scheduler.before_train_forward(*self.runtime_info)
        out = self.architecture(x)
        if self.training:
            if self.runtime_info is not None:
                extra_dict = self.scheduler.after_train_forward(
                    *self.runtime_info)
            return out, extra_dict['flops_loss']
        else:
            return out

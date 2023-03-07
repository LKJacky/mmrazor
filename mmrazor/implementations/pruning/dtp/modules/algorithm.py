# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.structures import BaseDataElement

from mmrazor.models import BaseAlgorithm
from mmrazor.registry import MODEL_WRAPPERS, MODELS, TASK_UTILS
from mmrazor.utils import RuntimeInfo, print_log
from .mutator import ImpMutator
from .scheduler import BaseDTPScheduler


def convert_float_to_tenosr(res: dict, device):
    for k in res:
        if not isinstance(res[k], torch.Tensor):
            res[k] = torch.tensor(res[k], device=device)
    return res


@MODELS.register_module()
class BaseDTPAlgorithm(BaseAlgorithm):

    def __init__(
            self,
            architecture: Union[BaseModel, Dict],
            mutator_cfg=dict(
                type='ImpMutator',
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
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.mutator: ImpMutator = MODELS.build(mutator_cfg)
        scheduler.update(dict(mutator=self.mutator, model=self.architecture))
        self.scheduler: BaseDTPScheduler = TASK_UTILS.build(scheduler)

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


@MODELS.register_module()
class DTPAlgorithm(BaseAlgorithm):

    def __init__(
            self,
            architecture: Union[BaseModel, Dict],
            mutator_cfg=dict(
                type='ImpMutator',
                channel_unit_cfg=dict(
                    type='ImpUnit',
                    default_args=dict(
                        imp_type='dtp',
                        grad_clip=-1,
                    )),
                parse_cfg=dict(
                    _scope_='mmrazor',
                    type='ChannelAnalyzer',
                    demo_input=dict(
                        type='DefaultDemoInput',
                        input_shape=[1, 3, 224, 224],
                    ),
                    tracer_type='FxTracer'),
            ),
            target_flop=0.5,
            flop_loss_weight=1.0,
            prune_iter_ratio=0.6,
            update_ratio=0.7,
            #
            data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
            init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.target_flop = target_flop
        self.flop_loss_weight = flop_loss_weight
        self.prune_iter_ratio = prune_iter_ratio
        self.update_ratio = update_ratio

        self.mutator: ImpMutator = MODELS.build(mutator_cfg)
        self.mutator.prepare_from_supernet(self.architecture)

        self.original_flops = self.mutator.get_soft_flop(
            self.architecture).detach().item()
        print_log(f'Get init flops {self.original_flops/1e6}')

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor'):
        # update

        if self.training:
            if RuntimeInfo.iter() == 0:
                self.mutator.resort()
                print_log('Resorted')

        res: dict = super().forward(inputs, data_samples, mode)  # type: ignore

        if self.training:
            if RuntimeInfo().iter(
            ) < RuntimeInfo.max_iters() * self.update_ratio:

                if mode == 'loss':
                    # flop_loss
                    if self.current_target > 0:
                        current_flops = self.mutator.get_soft_flop(
                            self.architecture)
                        res['flop_loss'] = self.flop_loss(
                            current_flops) * self.flop_loss_weight
                        res['soft_flop'] = current_flops.detach()
                        res['target'] = torch.tensor(self.current_target)
                    else:
                        pass
            else:
                self.mutator.requires_grad_(False)
        return res

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper) -> Dict[str, torch.Tensor]:
        self._before_train_step()
        res = super().train_step(data, optim_wrapper)
        self._after_train_step()
        return res

    def _before_train_step(self):
        self.mutator.save_info()

    def _after_train_step(self):
        self.mutator.limit_value()
        with torch.no_grad():
            if self.mutator.units[0].imp_type == 'dtp' or self.mutator.units[
                    0].imp_type == 'dtp_a':
                if RuntimeInfo().iter(
                ) < RuntimeInfo().max_iters() * self.update_ratio:
                    if self.current_target == self.target_flop:
                        self.mutator.adjust_to_target(
                            self.architecture,
                            self.target_flop * self.original_flops,
                            self.original_flops)

    #

    def flop_loss(self, current_flop):
        return (current_flop / self.original_flops - self.current_target)**2

    @property
    def current_target(self):
        iter = RuntimeInfo().iter()
        total_iter = int(self.prune_iter_ratio * RuntimeInfo().max_iters())
        if iter > total_iter:
            return self.target_flop
        else:
            return 1 - (1 - self.target_flop) * (iter / total_iter)


@MODEL_WRAPPERS.register_module()
class DTPAlgorithmDDP(MMDistributedDataParallel):
    """Train step for group fisher."""

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper) -> Dict[str, torch.Tensor]:
        DTPAlgorithm._before_train_step(self.module)
        res = super().train_step(data, optim_wrapper)
        DTPAlgorithm._after_train_step(self.module)
        return res

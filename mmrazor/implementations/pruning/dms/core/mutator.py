# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn

from mmrazor.models.mutators.base_mutator import BaseMutator
from mmrazor.models.task_modules.demo_inputs import DefaultDemoInput
from mmrazor.registry import MODELS, TASK_UTILS
from ...dtp.modules.dtp_adaptive import DTPAMutator
from ...dtp.modules.ops import QuickFlopMixin
from .mobilenet import MobileNetLayers
from .mutable import BlockThreshold, MutableBlocks
from .op import DynamicStage
from .resnet import ResLayer
from .resnet_img import ResLayer as ResLayerImg


def replace_modules(model: nn.Module, module_map={}):

    def replace_op(model: nn.Module, name: str, module: nn.Module):
        names = name.split('.')
        for sub_name in names[:-1]:
            model = getattr(model, sub_name)

        setattr(model, names[-1], module)

    for name, module in model.named_modules():
        if type(module) in module_map:
            new_module = module_map[type(module)].convert_from(module)
            replace_op(model, name, new_module)


class BlockInitialer:

    def __init__(
            self,
            dynamic_statge_module=DynamicStage,
            block_mixin_layers=[ResLayer, ResLayerImg,
                                MobileNetLayers]) -> None:
        self.dyanmic_stage_module = dynamic_statge_module
        self.block_mixin_layers = block_mixin_layers
        self.stages: List[DynamicStage] = []

    def prepare_from_supernet(self, supernet: nn.Module) -> List:
        map_dict = {}
        for block in self.block_mixin_layers:
            map_dict[block] = self.dyanmic_stage_module
        replace_modules(
            supernet,
            module_map=map_dict,
        )
        mutables = []
        for module in supernet.modules():
            if isinstance(module, self.dyanmic_stage_module):
                mutables.append(module.mutable_blocks)
                self.stages.append(module)
                module.mutable_blocks.requires_grad_(True)
        return mutables


def to_hard(scale):
    hard = (scale >= BlockThreshold).float()
    return hard.detach() - scale.detach() + scale


@MODELS.register_module()
class DMSMutator(BaseMutator):

    def __init__(self,
                 dtp_mutator_cfg=dict(
                     type='DTPAMutator',
                     channel_unit_cfg=dict(
                         type='DTPTUnit', default_args=dict()),
                     parse_cfg=dict(
                         _scope_='mmrazor',
                         type='ChannelAnalyzer',
                         demo_input=dict(
                             type='DefaultDemoInput',
                             input_shape=(1, 3, 32, 32),
                         ),
                         tracer_type='FxTracer'),
                 ),
                 init_cfg=None) -> None:
        super().__init__(init_cfg)

        self.dtp_mutator: DTPAMutator = MODELS.build(dtp_mutator_cfg)
        self.block_initializer = BlockInitialer()
        self.block_mutables: List[MutableBlocks] = nn.ModuleList()

    def prepare_from_supernet(self, supernet) -> None:
        self.saved_model = [supernet]
        self.dtp_mutator.prepare_from_supernet(supernet)
        self.block_mutables = nn.ModuleList(
            self.block_initializer.prepare_from_supernet(supernet))

    def info(self):

        mutable_info = ''

        @torch.no_grad()
        def stage_info(stage: DynamicStage):
            scales = ''
            for block in stage.removable_block:
                scales += f'{block.flop_scale:.2f} '
            return scales

        for mut, stage in zip(self.block_mutables,
                              self.block_initializer.stages):
            mutable_info += mut.info() + '\tratio:\t' + stage_info(
                stage) + '\n'

        flop_info = f'soft_flop: {self.get_soft_flop(self.saved_model[0])/1e6}'

        return self.dtp_mutator.info() + '\n' + mutable_info + '\n' + flop_info

    @torch.no_grad()
    def init_quick_flop(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, QuickFlopMixin):
                module.quick_flop_start_record()
        demo_input: DefaultDemoInput = TASK_UTILS.build(
            self.dtp_mutator.demo_input)
        model.eval()
        input = demo_input.get_data(model, training=False)
        if isinstance(input, dict):
            input['mode'] = 'tensor'
            model(**input)
        else:
            model(input)
        for module in model.modules():
            if isinstance(module, QuickFlopMixin):
                module.quick_flop_end_record()

    @torch.no_grad()
    def limit_value(self):
        self.dtp_mutator.limit_value()
        for mul in self.block_mutables:
            mul.limit_value()

    def get_soft_flop(self, model):
        return QuickFlopMixin.get_flop(model)

    def channel_depth_train(self):
        self.dtp_mutator.ratio_train()
        for mul in self.block_mutables:
            mul.requires_grad_(True)
        self.set_soft_flop_scale_converter(None)

    def channel_train(self):
        self.dtp_mutator.ratio_train()
        for mul in self.block_mutables:
            mul.requires_grad_(False)
        self.set_soft_flop_scale_converter(to_hard)

    @property
    def choice_template(self):
        return self.dtp_mutator.choice_template

    # inherit from BaseMutator
    def search_groups(self):
        return super().search_groups

    def set_soft_flop_scale_converter(self, fun):
        for mut in self.block_mutables:
            mut.flop_scale_converter = fun

from typing import Dict, Optional, Union
from mmengine.model import BaseModel
import torch
from mmrazor.implementations.pruning.dms.core.algorithm import BaseDTPAlgorithm, BaseAlgorithm, DmsAlgorithmMixin, update_dict_reverse
from mmrazor.implementations.pruning.dms.core.op import ImpLinear
from mmrazor.implementations.pruning.dms.core.dtp import DTPAMutator
from mmrazor.implementations.pruning.dms.core.models.opt.opt_analyzer import OutChannel, InChannel
from mmrazor.models.mutables import SequentialMutableChannelUnit

from mmcls.models.backbones.swin_transformer import SwinBlockSequence, SwinBlock, ShiftWindowMSA, SwinTransformer, PatchMerging
from mmcls.models.heads.vision_transformer_head import VisionTransformerClsHead
from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from mmrazor.utils import print_log
import json
from mmcls.registry import MODELS as CLSMODELS
from mmrazor.registry import MODELS
import torch.nn as nn
import copy
import mmcv
from mmcv.cnn.bricks.wrappers import Linear as MMCVLinear
from mmrazor.models.architectures.dynamic_ops import DynamicLinear
import numpy as np
from mmrazor.implementations.pruning.dms.core.op import (ImpModuleMixin,
                                                         DynamicBlockMixin,
                                                         MutableAttn,
                                                         QuickFlopMixin,
                                                         ImpLinear)
from mmrazor.implementations.pruning.dms.core.mutable import (
    ImpMutableChannelContainer, MutableChannelForHead, MutableChannelWithHead,
    MutableHead)
from dms_deit import MMCVLinear
# mutator ####################################################################################


class SwinAnalyzer:

    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def parse_block(cls, block: SwinBlock, prefix=''):
        mlp_unit = SequentialMutableChannelUnit(
            block.ffn.layers[0][0].out_features)
        mlp_unit.add_output_related(
            OutChannel(prefix + 'ffn.layers.0.0', block.ffn.layers[0][0]))

        mlp_unit.add_input_related(
            InChannel(prefix + 'ffn.layers.1', block.ffn.layers[1]))
        return mlp_unit.config_template(
            with_channels=True, with_init_args=True)

    @classmethod
    def parse_staget(cls, stage: SwinBlockSequence, prefix=''):
        unit = SequentialMutableChannelUnit(
            stage.blocks[0].norm1.normalized_shape[-1])

        def parse_block(block: SwinBlock, prefix):
            unit.add_input_related(InChannel(prefix + 'norm1', block.norm1))
            unit.add_input_related(
                InChannel(prefix + 'attn.w_msa.qkv', block.attn.w_msa.qkv))
            unit.add_input_related(InChannel(prefix + 'norm2', block.norm2))
            unit.add_input_related(
                InChannel(prefix + 'attn.ffn.layers.0.0',
                          block.ffn.layers[0][0]))

            unit.add_output_related(
                OutChannel(prefix + 'attn.w_msa.proj', block.attn.w_msa.proj))
            unit.add_output_related(
                OutChannel(prefix + 'ffn.layers.1', block.ffn.layers[1]))

            return unit

        for name, block in stage.blocks.named_children():
            parse_block(block, prefix=f'{prefix}blocks.{name}.')

        if stage.downsample is not None:
            unit.add_input_related(
                InChannel(prefix + 'downsample.reduction',
                          stage.downsample.reduction))
            unit.add_input_related(
                InChannel(prefix + 'downsample.norm', stage.downsample.norm))
        return unit

    def get_config(self):
        config = {}
        model = self.model
        backbone: SwinTransformer = self.model.backbone

        ## mlp
        for name, stage in backbone.stages.named_children():

            for name_block, block in stage.blocks.named_children():
                config[f'backbone.{name}.{name_block}.ffn'] = self.parse_block(
                    block, f'backbone.stages.{name}.blocks.{name_block}.')
        # stages
        for name, stage in backbone.stages.named_children():
            unit = self.parse_staget(stage, f'backbone.stages.{name}.')
            if name == '0':
                unit.add_output_related(
                    OutChannel('backbone.patch_embed.projection',
                               backbone.patch_embed.projection))
                unit.add_output_related(
                    OutChannel('backbone.patch_embed.norm',
                               backbone.patch_embed.norm))
            else:
                unit.add_output_related(
                    OutChannel(
                        f'backbone.stages.{int(name)-1}.downsample.reduction',
                        backbone.stages[int(name) - 1].downsample.reduction))

            if name == '3':
                unit.add_input_related(
                    InChannel('backbone.norm3', backbone.norm3))
                unit.add_input_related(InChannel('head.fc', model.head.fc))
            config[f'stage{name}'] = unit.config_template(
                with_channels=True, with_init_args=True)

        config = self.post_process(config)
        import json
        print(json.dumps(config, indent=4))
        return config

    @classmethod
    def post_process(cls, config: dict):
        for unit_name in config:
            for key in copy.copy(config[unit_name]['init_args']):
                if key != 'num_channels':
                    config[unit_name]['init_args'].pop(key)
            config[unit_name].pop('choice')
        return config


@MODELS.register_module()
class SwinMutator(DTPAMutator):

    def prepare_from_supernet(self, supernet) -> None:

        analyzer = SwinAnalyzer(supernet)
        config = analyzer.get_config()

        units = self._prepare_from_unit_cfg(supernet, config)
        for unit in units:
            unit.prepare_for_pruning(supernet)
            self._name2unit[unit.name] = unit
        self.units = nn.ModuleList(units)

        for unit in self.mutable_units:
            unit.requires_grad_(True)
        assert len(self.mutable_units) > 0
        for unit in self.units:
            for channel in unit.input_related + unit.output_related:
                assert isinstance(channel, nn.Module)


@MODELS.register_module()
class SwinDms(BaseDTPAlgorithm):

    def __init__(self,
                 architecture: BaseModel,
                 mutator_cfg={},
                 scheduler={},
                 data_preprocessor=None,
                 init_cfg=None) -> None:
        BaseAlgorithm.__init__(self, architecture, data_preprocessor, init_cfg)
        model = self.architecture
        # model.backbone.layers = DeitLayers.convert_from(model.backbone.layers)

        default_mutator_kwargs = dict(
            prune_qkv=True,
            prune_block=True,
            dtp_mutator_cfg=dict(
                type='SwinMutator',
                channel_unit_cfg=dict(
                    type='DTPTUnit',
                    default_args=dict(extra_mapping={
                        MMCVLinear: ImpLinear,
                    })),
                parse_cfg=dict(
                    _scope_='mmrazor',
                    type='ChannelAnalyzer',
                    demo_input=dict(
                        type='DefaultDemoInput',
                        input_shape=(1, 3, 224, 224),
                    ),
                    tracer_type='FxTracer'),
            ),
            extra_module_mapping={
                # MultiheadAttention: DynamicAttention,
                # SplitAttention: DynamicAttention,
            },
            block_initilizer_kwargs=dict(
                stage_mixin_layers=[],
                dynamic_block_mapping={
                    # TransformerEncoderLayer: DyTransformerEncoderLayer
                }),
        )
        default_scheduler_kargs = dict(
            flops_target=0.8,
            decay_ratio=0.8,
            refine_ratio=0.2,
            flop_loss_weight=100,
            structure_log_interval=1000,
            by_epoch=True,
            target_scheduler='cos',
        )
        default_mutator_kwargs = update_dict_reverse(default_mutator_kwargs,
                                                     mutator_cfg)
        default_scheduler_kargs = update_dict_reverse(default_scheduler_kargs,
                                                      scheduler)
        DmsAlgorithmMixin.__init__(
            self,
            self.architecture,
            mutator_kwargs=default_mutator_kwargs,
            scheduler_kargs=default_scheduler_kargs)


# test ####################################################################################

if __name__ == '__main__':
    model_dict = dict(
        # model settings
        type='ImageClassifier',
        backbone=dict(
            type='SwinTransformer',
            arch='tiny',
            img_size=224,
            drop_path_rate=0.2),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=768,
            init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
            loss=dict(
                type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
            cal_acc=False),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ],
        train_cfg=dict(augments=[
            dict(type='Mixup', alpha=0.8),
            dict(type='CutMix', alpha=1.0)
        ]),
    )
    model = CLSMODELS.build(model_dict)

    alg = SwinDms(model)
    print(alg)
    print(alg.mutator.info())
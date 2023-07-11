import torch
from mmrazor.implementations.pruning.dms.core.algorithm import DmsGeneralAlgorithm
from mmrazor.implementations.pruning.dms.core.op import ImpLinear
from mmrazor.implementations.pruning.dms.core.dtp import DTPAMutator
from mmrazor.implementations.pruning.dms.core.models.opt.opt_analyzer import OutChannel, InChannel
from mmrazor.models.mutables import SequentialMutableChannelUnit
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer, VisionTransformer
from mmcls.models.heads.vision_transformer_head import VisionTransformerClsHead

from mmcls.registry import MODELS as CLS_MODELS
from mmrazor.registry import MODELS
import torch.nn as nn
import copy
import mmcv
from mmcv.cnn.bricks.wrappers import Linear as MMCVLinear
from mmrazor.models.architectures.dynamic_ops import DynamicLinear
import numpy as np

# mutator ####################################################################################


class DeitAnalyzer:

    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def parse_block(cls, block: TransformerEncoderLayer, prefix=''):
        mlp_unit = SequentialMutableChannelUnit(
            block.ffn.layers[0][0].out_features)
        mlp_unit.add_output_related(
            OutChannel(prefix + '.ffn.layers.0.0', block.ffn.layers[0][0]))

        mlp_unit.add_input_related(
            InChannel(prefix + '.ffn.layers.1', block.ffn.layers[1]))
        return mlp_unit.config_template(
            with_channels=True, with_init_args=True)

    @classmethod
    def parse_res_structure(cls, model):
        backbone: VisionTransformer = model.backbone
        head: VisionTransformerClsHead = model.head

        unit = SequentialMutableChannelUnit(
            backbone.patch_embed.projection.out_channels)

        unit.add_output_related(
            OutChannel(f"backbone.patch_embed.projection",
                       backbone.patch_embed.projection))
        unit.add_output_related(OutChannel(f"backbone.ln1", backbone.ln1))
        for name, block in backbone.layers.named_children():
            block: TransformerEncoderLayer
            unit.add_output_related(
                OutChannel(f"backbone.layers.{name}.ln1", block.ln1))
            unit.add_output_related(
                OutChannel(f"backbone.layers.{name}.ln2", block.ln2))

            unit.add_output_related(
                OutChannel(f"backbone.layers.{name}.attn.proj",
                           block.attn.proj))
            unit.add_output_related(
                OutChannel(f"backbone.layers.{name}.ffn.layers.1",
                           block.ffn.layers[1]))

        unit.add_input_related(InChannel(f"backbone.ln1", backbone.ln1))
        unit.add_input_related(InChannel(f"backbone.ln1", backbone.ln1))
        for name, block in backbone.layers.named_children():
            block: TransformerEncoderLayer
            unit.add_input_related(
                InChannel(f"backbone.layers.{name}.ln1", block.ln1))
            unit.add_input_related(
                InChannel(f"backbone.layers.{name}.ln2", block.ln2))

            unit.add_input_related(
                InChannel(f"backbone.layers.{name}.attn.qkv", block.attn.qkv))
            unit.add_input_related(
                InChannel(f"backbone.layers.{name}.ffn.layers.0.0",
                          block.ffn.layers[0][0]))
        unit.add_input_related(InChannel(f"head.layers.head", head.layers[0]))
        return unit.config_template(True, True)

    def get_config(self):
        config = {}
        backbone: VisionTransformer = self.model.backbone

        ## mlp
        for name, block in backbone.layers.named_children():
            config[f'backbone.layers.{name}.mlp'] = self.parse_block(
                block, f'backbone.layers.{name}')

        # res
        config['res'] = self.parse_res_structure(self.model)

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
class DeitMutator(DTPAMutator):

    def prepare_from_supernet(self, supernet) -> None:

        analyzer = DeitAnalyzer(supernet)
        config = analyzer.get_config()

        units = self._prepare_from_unit_cfg(supernet, config)
        for unit in units:
            unit.prepare_for_pruning(supernet)
            self._name2unit[unit.name] = unit
        self.units = nn.ModuleList(units)

        for unit in self.mutable_units:
            unit.requires_grad_(True)


class DeitDms(DmsGeneralAlgorithm):

    def to_static_model(self):
        model = super().to_static_model()
        backbone: VisionTransformer = model.backbone
        mask = self.mutator.dtp_mutator.mutable_units[
            -1].mutable_channel.mask.bool()
        backbone.cls_token = nn.Parameter(backbone.cls_token[:, :, mask])
        backbone.pos_embed = nn.Parameter(backbone.pos_embed[:, :, mask])
        return model


if __name__ == '__main__':
    model_dict = dict(
        type='ImageClassifier',
        backbone=dict(
            type='VisionTransformer',
            arch='deit-tiny',
            img_size=224,
            patch_size=16),
        neck=None,
        head=dict(
            type='VisionTransformerClsHead',
            num_classes=1000,
            in_channels=192,
            loss=dict(
                type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        ),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=.02),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
        ],
        train_cfg=dict(augments=[
            dict(type='Mixup', alpha=0.8),
            dict(type='CutMix', alpha=1.0)
        ]),
    )
    model = CLS_MODELS.build(model_dict)
    print(model)

    algo = DeitDms(
        model,
        mutator_kwargs=dict(
            prune_qkv=False,
            prune_block=False,
            dtp_mutator_cfg=dict(
                type='DeitMutator',
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
            extra_module_mapping={}),
    )
    print(algo.mutator.info())

    def rand_mask(mask):
        return (torch.rand_like(mask) < 0.5).float()

    for unit in algo.mutator.dtp_mutator.mutable_units:                                                        \

        unit.mutable_channel.mask.data = rand_mask(unit.mutable_channel.mask)
    model = algo.to_static_model()
    x = torch.rand([1, 3, 224, 224])
    print(model)
    model(x)

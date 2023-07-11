import torch
from mmrazor.implementations.pruning.dms.core.algorithm import DmsGeneralAlgorithm
from mmrazor.implementations.pruning.dms.core.op import ImpLinear
from mmrazor.implementations.pruning.dms.core.dtp import DTPAMutator
from mmrazor.implementations.pruning.dms.core.models.opt.opt_analyzer import OutChannel, InChannel
from mmrazor.models.mutables import SequentialMutableChannelUnit
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer, VisionTransformer, MultiheadAttention
from mmcls.models.heads.vision_transformer_head import VisionTransformerClsHead
from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin

from mmcls.registry import MODELS as CLS_MODELS
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
# ops ####################################################################################


class SplitAttention(MultiheadAttention):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        qkv_bias = self.qkv.bias is not None
        self.q = nn.Linear(self.input_dims, self.embed_dims, bias=qkv_bias)
        self.k = nn.Linear(self.input_dims, self.embed_dims, bias=qkv_bias)
        self.v = nn.Linear(self.input_dims, self.embed_dims, bias=qkv_bias)
        delattr(self, 'qkv')

        self.init_kargs = kwargs

    def forward(self, x):
        B, N, _ = x.shape

        q, k, v = self.q(x), self.k(x), self.v(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x

    @classmethod
    def convert_from(cls, attn: MultiheadAttention):
        module = SplitAttention(
            embed_dims=attn.embed_dims,
            num_heads=attn.num_heads,
            input_dims=attn.input_dims,
            attn_drop=attn.attn_drop.p,
            proj_drop=attn.proj_drop.p,
            dropout_layer=dict(type='Dropout', drop_prob=0.),
            qkv_bias=attn.qkv.bias is not None,
            qk_scale=attn.scale,
            proj_bias=attn.proj.bias is not None,
            v_shortcut=attn.v_shortcut,
            use_layer_scale=not isinstance(attn.gamma1, nn.Identity),
        )
        return module


class DynamicAttention(SplitAttention, DynamicChannelMixin, MutableAttn,
                       QuickFlopMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MutableAttn.__init__(self)
        QuickFlopMixin.__init__(self)

        self.init_args = args
        self.init_kwargs = kwargs
        self.mutable_attrs = nn.ModuleDict()

        self.q = ImpLinear.convert_from(self.q)
        self.k = ImpLinear.convert_from(self.k)
        self.v = ImpLinear.convert_from(self.v)

        self.proj = ImpLinear.convert_from(self.proj)

        self.in_channels = self.q.in_features
        self.out_channels = self.proj.out_features

    def register_mutable_attr(self, attr: str, mutable):
        if attr == 'in_channels':
            self.q.register_mutable_attr(attr, mutable)
            self.k.register_mutable_attr(attr, mutable)
            self.v.register_mutable_attr(attr, mutable)
            # self.qkv
        elif attr == 'out_channels':
            self.proj.register_mutable_attr(attr, mutable)
        else:
            raise NotImplementedError()
        self.mutable_attrs[attr] = mutable

    @classmethod
    def convert_from(cls, module: SplitAttention):
        if isinstance(module, MultiheadAttention):
            module = SplitAttention.convert_from(module)
        new_module = cls(**module.init_kargs)
        new_module.load_state_dict(module.state_dict(), strict=False)
        return new_module

    @property
    def static_op_factory(self):
        return SplitAttention

    def init_mutables(self):
        m_head = MutableHead(self.num_heads)
        m_qk = MutableChannelForHead(self.q.out_features, self.num_heads)
        m_v = MutableChannelForHead(self.v.out_features, self.num_heads)
        mutable_qk = MutableChannelWithHead(m_head, m_qk)
        mutable_v = MutableChannelWithHead(m_head, m_v)

        try:
            self.q.register_mutable_attr(
                'out_channels',
                ImpMutableChannelContainer(self.q.out_features))
            self.k.register_mutable_attr(
                'out_channels',
                ImpMutableChannelContainer(self.k.out_features))
            self.v.register_mutable_attr(
                'out_channels',
                ImpMutableChannelContainer(self.v.out_features))
            self.proj.register_mutable_attr(
                'in_channels',
                ImpMutableChannelContainer(self.proj.in_features))
        except Exception:
            pass
        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.q, mutable=mutable_qk, is_to_output_channel=True)
        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.k, mutable=mutable_qk, is_to_output_channel=True)
        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.v, mutable=mutable_v, is_to_output_channel=True)

        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.proj, mutable_v, is_to_output_channel=False)

        self.attn_mutables = {'head': m_head, 'qk': m_qk, 'v': m_v}

        return m_head, m_qk, m_v

    def to_static_op(self):

        if 'head' in self.attn_mutables:
            num_heads = int(self.attn_mutables['head'].mask.sum().item())
        else:
            num_heads = self.num_heads

        module: SplitAttention = SplitAttention(*self.init_args,
                                                **self.init_kwargs)
        module.q = self.q.to_static_op()
        module.k = self.k.to_static_op()
        module.v = self.v.to_static_op()
        module.proj = self.proj.to_static_op()
        module.num_heads = num_heads
        module.head_dims = module.q.out_features // num_heads
        module.embed_dims = module.v.out_features
        return module

    def soft_flop(self):
        flops = 0
        flops = flops + QuickFlopMixin.get_flop(self.q)
        flops = flops + QuickFlopMixin.get_flop(self.k)
        flops = flops + QuickFlopMixin.get_flop(self.v)
        flops = flops + QuickFlopMixin.get_flop(self.proj)

        mutable_head: MutableHead = self.attn_mutables['head']
        head = mutable_head.current_imp_flop.sum()
        B, N, _ = self.quick_flop_recorded_in_shape[0]

        flops = flops + B * head * N * N * self.head_dims * 2
        return flops


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
                InChannel(f"backbone.layers.{name}.attn.q", block.attn.q))
            unit.add_input_related(
                InChannel(f"backbone.layers.{name}.attn.k", block.attn.k))
            unit.add_input_related(
                InChannel(f"backbone.layers.{name}.attn.v", block.attn.v))
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
            extra_module_mapping={MultiheadAttention: DynamicAttention}),
    )
    print(algo.mutator.info())

    def rand_mask(mask):
        while True:
            mask = (torch.rand_like(mask) < 0.5).float()
            if mask.sum() != 0:
                break
        return mask

    for unit in algo.mutator.dtp_mutator.mutable_units:
        unit.mutable_channel.mask.data = rand_mask(unit.mutable_channel.mask)
    for attn_mutables in algo.mutator.attn_mutables:
        head_mutable = attn_mutables['head']
        q_mutable = attn_mutables['qk']
        kv_mutable = attn_mutables['v']
        head_mutable.mask = rand_mask(head_mutable.mask)
        q_mutable.mask = rand_mask(q_mutable.mask)
        kv_mutable.mask = rand_mask(kv_mutable.mask)

    model = algo.to_static_model()
    x = torch.rand([1, 3, 224, 224])
    print(model)
    model(x)

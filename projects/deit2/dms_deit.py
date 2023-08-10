from typing import Dict, Optional, Union
from mmengine.model import BaseModel
import torch
from mmrazor.implementations.pruning.dms.core.algorithm import BaseDTPAlgorithm, BaseAlgorithm, DmsAlgorithmMixin, update_dict_reverse
from mmrazor.implementations.pruning.dms.core.op import ImpLinear
from mmrazor.implementations.pruning.dms.core.dtp import DTPAMutator
from mmrazor.implementations.pruning.dms.core.models.opt.opt_analyzer import OutChannel, InChannel
from mmrazor.models.mutables import SequentialMutableChannelUnit
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer, VisionTransformer, MultiheadAttention
from mmcls.models.heads.vision_transformer_head import VisionTransformerClsHead
from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from mmrazor.utils import print_log
import json
from mmcls.registry import MODELS as CLS_MODELS
from mmrazor.registry import MODELS
import torch.nn as nn
import copy
import mmcv
from mmcv.cnn.bricks.wrappers import Linear as MMCVLinear
from mmrazor.models.architectures.dynamic_ops import DynamicLinear
import numpy as np
from mmrazor.implementations.pruning.dms.core.op import (
    ImpModuleMixin, DynamicBlockMixin, MutableAttn, QuickFlopMixin, ImpLinear,
    DynamicStage)
from mmrazor.implementations.pruning.dms.core.mutable import (
    ImpMutableChannelContainer, MutableChannelForHead, MutableChannelWithHead,
    MutableHead)
from mmrazor.implementations.pruning.dms.core.mutable import DMSMutableMixIn

# ops ####################################################################################


class MyMultiheadAttention(MultiheadAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qk_dim = self.qkv.out_features // 3
        self.v_dim = self.qkv.out_features // 3

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv[:, :, :self.qk_dim], qkv[:, :, self.qk_dim:self.qk_dim *
                                               2], qkv[:, :, self.qk_dim * 2:]

        def reshape(x: torch.Tensor):
            return x.reshape([B, N, self.num_heads, -1]).permute(0, 2, 1, 3)

        q, k, v = map(reshape, [q, k, v])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class DynamicAttention(MultiheadAttention, DynamicChannelMixin, MutableAttn,
                       QuickFlopMixin):

    def __init__(self, *args, **kwargs):
        MultiheadAttention.__init__(self, *args, **kwargs)
        MutableAttn.__init__(self)
        QuickFlopMixin.__init__(self)

        self.init_args = args
        self.init_kwargs = kwargs
        self.mutable_attrs = nn.ModuleDict()

        self.qkv = ImpLinear.convert_from(self.qkv)
        self.proj = ImpLinear.convert_from(self.proj)

        self.in_channels = self.qkv.in_features
        self.out_channels = self.proj.out_features

    def register_mutable_attr(self, attr: str, mutable):
        if attr == 'in_channels':
            self.qkv.register_mutable_attr(attr, mutable)
            # self.qkv
        elif attr == 'out_channels':
            self.proj.register_mutable_attr(attr, mutable)
        else:
            raise NotImplementedError()
        self.mutable_attrs[attr] = mutable

    @classmethod
    def convert_from(cls, attn: MultiheadAttention):
        new_module = cls(
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
        new_module.out_drop = attn.out_drop
        new_module.load_state_dict(attn.state_dict(), strict=True)
        return new_module

    @property
    def static_op_factory(self):
        return MultiheadAttention

    def init_mutables(self):
        m_head = MutableHead(self.num_heads)
        m_qk = MutableChannelForHead(self.qkv.out_features // 3,
                                     self.num_heads)
        m_v = MutableChannelForHead(self.qkv.out_features // 3, self.num_heads)
        mutable_qk = MutableChannelWithHead(m_head, m_qk)
        mutable_v = MutableChannelWithHead(m_head, m_v)

        try:
            self.qkv.register_mutable_attr(
                'out_channels',
                ImpMutableChannelContainer(self.qkv.out_features))
            self.proj.register_mutable_attr(
                'in_channels',
                ImpMutableChannelContainer(self.proj.in_features))
            ##
            self.qkv.register_mutable_attr(
                'in_channels',
                ImpMutableChannelContainer(self.qkv.in_features))
            self.proj.register_mutable_attr(
                'out_channels',
                ImpMutableChannelContainer(self.proj.out_features))
        except Exception:
            pass
        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.qkv,
            mutable=mutable_qk,
            is_to_output_channel=True,
            start=0,
            end=self.qkv.out_features // 3)
        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.qkv,
            mutable=mutable_qk,
            is_to_output_channel=True,
            start=self.qkv.out_features // 3 * 1,
            end=self.qkv.out_features // 3 * 2)
        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.qkv,
            mutable=mutable_v,
            is_to_output_channel=True,
            start=self.qkv.out_features // 3 * 2,
            end=self.qkv.out_features // 3 * 3)

        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.proj, mutable_v, is_to_output_channel=False)

        self.attn_mutables = {'head': m_head, 'qk': m_qk, 'v': m_v}
        self.qkv.use_out_imp = True
        return m_head, m_qk, m_v

    def to_static_op(self):

        if 'head' in self.attn_mutables:
            num_heads = int(self.attn_mutables['head'].mask.sum().item())
        else:
            num_heads = self.num_heads
        head_mask = self.attn_mutables['head'].mask.bool()
        qk_dim = int(self.attn_mutables['qk'].mask[head_mask].sum().item() //
                     num_heads)
        v_dim = int(self.attn_mutables['v'].mask[head_mask].sum().item() //
                    num_heads)

        module: MyMultiheadAttention = MyMultiheadAttention(
            *self.init_args, **self.init_kwargs)
        module.qkv = self.qkv.to_static_op()
        module.proj = self.proj.to_static_op()
        module.num_heads = num_heads
        module.head_dims = qk_dim
        module.embed_dims = v_dim
        module.out_drop = self.out_drop

        module.qk_dim = qk_dim * num_heads
        module.v_dim = v_dim * num_heads

        module.scale = (module.qk_dim // num_heads)**-0.5
        assert module.qkv.out_features == (qk_dim * 2 + v_dim) * num_heads

        return module

    def soft_flop(self):
        flops = 0
        flops = flops + QuickFlopMixin.get_flop(self.qkv)

        flops = flops + QuickFlopMixin.get_flop(self.proj)

        mutable_head: MutableHead = self.attn_mutables['head']
        mutable_qk: MutableChannelForHead = self.attn_mutables['qk']
        mutable_v: MutableChannelForHead = self.attn_mutables['v']
        head = mutable_head.current_imp_flop.sum()
        qk_dim = mutable_qk.current_imp_flop.sum() / head
        v_dim = mutable_v.current_imp_flop.sum() / head
        B, N, _ = self.quick_flop_recorded_in_shape[0]

        flops = flops + B * head * N * N * (qk_dim + v_dim)
        return flops


class DeitLayers(nn.ModuleList):

    @classmethod
    def convert_from(cls, module):
        new = cls()
        for m in module:
            new.append(m)
        return new


class DeitDynamicStage(DynamicStage):

    def to_static_op(self):
        op = super().to_static_op()
        from mmengine.model.base_module import ModuleList
        new = ModuleList()
        for m in op:
            new.append(m)
        return new


class DyTransformerEncoderLayer(TransformerEncoderLayer, DynamicBlockMixin):

    def __init__(self, *args, **kwargs):
        TransformerEncoderLayer.__init__(self, *args, **kwargs)
        DynamicBlockMixin.__init__(self)
        self.init_args = args
        self.init_kwargs = kwargs

    @property
    def is_removable(self):
        return True

    @classmethod
    def convert_from(cls, module: TransformerEncoderLayer):
        new_module = DyTransformerEncoderLayer(
            embed_dims=module.embed_dims,
            num_heads=module.attn.num_heads,
            feedforward_channels=module.ffn.feedforward_channels,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            num_fcs=2,
            qkv_bias=True,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            init_cfg=None)
        new_module.embed_dims = module.embed_dims
        new_module.norm1_name = module.norm1_name
        new_module.norm2_name = module.norm2_name

        for name, child in module.named_children():
            setattr(new_module, name, child)
        return new_module

    def to_static_op(self):
        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static

        module = self.static_op_factory(*self.init_args, **self.init_kwargs)

        module.embed_dims = self.embed_dims
        module.norm1_name = self.norm1_name
        module.norm2_name = self.norm2_name

        for name, m in self.named_children():  # type: ignore
            assert hasattr(module, name)
            setattr(module, name, _dynamic_to_static(m))
        return module

    @property
    def static_op_factory(self):
        return TransformerEncoderLayer

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.scale
        x = x + self.ffn(self.norm2(x), identity=None) * self.scale
        return x

    def __repr__(self):
        return TransformerEncoderLayer.__repr__(self)


# schedulers ####################################################################################

from mmengine.optim.scheduler import LinearLR, CosineAnnealingLR
from mmengine.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class MyLinearLR(LinearLR):

    def __init__(self, optimizer, *args, mutator_lr=0.1, **kwargs):
        self.mutator_lr = mutator_lr
        super().__init__(optimizer, *args, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        res = super()._get_value()
        mutator_lr = self.mutator_lr
        for i in range(len(self.base_values)):
            if self.base_values[i] == mutator_lr:
                res[i] = mutator_lr
        return res


@PARAM_SCHEDULERS.register_module()
class MyCosineAnnealingLR(CosineAnnealingLR):

    def __init__(self, optimizer, *args, mutator_lr=0.1, **kwargs):
        self.mutator_lr = mutator_lr
        super().__init__(optimizer, *args, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        res = super()._get_value()
        mutator_lr = self.mutator_lr
        for i in range(len(self.base_values)):
            if self.base_values[i] == mutator_lr:
                res[i] = mutator_lr
        return res


# mutator ####################################################################################


@CLS_MODELS.register_module()
class VisionTransformer2(VisionTransformer):
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
        **dict.fromkeys(
            ['eva-g', 'eva-giant'],
            {
                # The implementation in EVA
                # <https://arxiv.org/abs/2211.07636>
                'embed_dims': 1408,
                'num_layers': 40,
                'num_heads': 16,
                'feedforward_channels': 6144
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'], {
                'embed_dims': 384,
                'num_layers': 16,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
    }


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


@MODELS.register_module()
def DeitSubModel(
    algorithm: dict,
    reset_params=False,
    **kargs,
):
    """Convert a algorithm(with an architecture) to a static pruned
    architecture.

    Args:
        algorithm (Union[BaseAlgorithm, dict]): The pruning algorithm to
            finetune.
        divisor (int): The divisor to make the channel number
            divisible. Defaults to 1.

    Returns:
        nn.Module: a static model.
    """
    # # init algorithm
    algorithm_dict = algorithm
    pruned = algorithm_dict.pop('pruned')
    if isinstance(algorithm, dict):
        algorithm = MODELS.build(algorithm)  # type: ignore
    assert isinstance(algorithm, DeitDms)
    state = torch.load(pruned, map_location='cpu')['state_dict']
    algorithm.load_state_dict(state)

    print_log(f"{algorithm.mutator.info()}")

    return algorithm.to_static_model(reset=reset_params)


@MODELS.register_module()
class DeitDms(BaseDTPAlgorithm):

    def __init__(self,
                 architecture: BaseModel,
                 mutator_cfg={},
                 scheduler={},
                 data_preprocessor=None,
                 init_cfg=None) -> None:
        BaseAlgorithm.__init__(self, architecture, data_preprocessor, init_cfg)
        model = self.architecture
        old_arch = copy.deepcopy(self.architecture)
        old_arch.init_weights()

        model.backbone.layers = DeitLayers.convert_from(model.backbone.layers)

        default_mutator_kwargs = dict(
            prune_qkv=True,
            prune_block=True,
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
            extra_module_mapping={
                MultiheadAttention: DynamicAttention,
            },
            block_initilizer_kwargs=dict(
                dynamic_statge_module=DeitDynamicStage,
                stage_mixin_layers=[DeitLayers],
                dynamic_block_mapping={
                    TransformerEncoderLayer: DyTransformerEncoderLayer
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
        self.architecture.load_state_dict(old_arch.state_dict())

    @torch.no_grad()
    def to_static_model(self, reset=True):
        model = super().to_static_model(reset_params=reset)
        backbone: VisionTransformer = model.backbone
        mask = self.mutator.dtp_mutator.mutable_units[
            -1].mutable_channel.mask.bool()
        # necessary for static model
        backbone.cls_token = nn.Parameter(backbone.cls_token[:, :, mask])
        backbone.pos_embed = nn.Parameter(backbone.pos_embed[:, :, mask])
        backbone.out_indices = [len(backbone.layers) - 1]

        print_log('Staic model')
        print_log(model)

        if reset:
            backbone.cls_token.data.fill_(0)
            backbone.pos_embed.data.fill_(0)
            backbone.patch_embed.projection.reset_parameters()

        return model


if __name__ == '__main__':

    SmallModelDict = dict(
        type='ImageClassifier',
        backbone=dict(
            type='VisionTransformer2',
            arch='deit-small',
            img_size=224,
            patch_size=16),
        neck=None,
        head=dict(
            type='VisionTransformerClsHead',
            num_classes=1000,
            in_channels=384,
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
    TinyModelDict = dict(
        type='ImageClassifier',
        backbone=dict(
            type='VisionTransformer2',
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

    def set_mutable(mutable: DMSMutableMixIn, ratio=0.5):
        mutable.e.data.fill_(ratio)
        mutable.taylor.data = torch.rand_like(mutable.taylor)
        mutable.sync_mask(fix_bug=True)

    def test_module(module: nn.Module(), x):
        module.train()
        x = nn.Parameter(x, requires_grad=True)
        y = module(x)
        loss = y.sum()
        loss.backward()
        return x, x.grad

    def test_two_modules(static_module, dy_module, shape):
        x = torch.rand(shape)
        y1, g1 = test_module(static_module, x)
        y2, g2 = test_module(dy_module, x)
        print('two module test:')
        print((y1 - y2).abs().max())
        print((g1 - g2).abs().max())

    def test_attn():
        attn = MultiheadAttention(embed_dims=320, num_heads=4, input_dims=320)
        dyattn = DynamicAttention.convert_from(attn)
        head, qk, v = dyattn.init_mutables()
        test_two_modules(attn, dyattn, [2, 32, 320])

        # to static
        attn = MultiheadAttention(embed_dims=80, num_heads=2, input_dims=320)
        attn.proj = nn.Linear(80, 320)
        set_mutable(head)
        set_mutable(qk)
        set_mutable(v)
        static_attn = dyattn.to_static_op()
        static_attn.load_state_dict(attn.state_dict())
        test_two_modules(attn, static_attn, [2, 32, 320])

    def test_model():
        model_dict = SmallModelDict
        model = CLS_MODELS.build(model_dict)

        algo = DeitDms(model, )
        print(algo.mutator.info())

        for unit in algo.mutator.dtp_mutator.mutable_units:
            set_mutable(unit.mutable_channel)
        for mutable in algo.mutator.block_mutables:
            set_mutable(mutable, ratio=11 / 16)
        for attn_mutables in algo.mutator.attn_mutables:
            head_mutable = attn_mutables['head']
            q_mutable = attn_mutables['qk']
            kv_mutable = attn_mutables['v']
            set_mutable(head_mutable)
            # set_mutable(q_mutable)
            # set_mutable(kv_mutable)
        print(algo.mutator.info())
        model = algo.to_static_model()
        x = torch.rand([1, 3, 224, 224])
        model(x)

        tiny: nn.Module = CLS_MODELS.build(TinyModelDict)
        print(model)
        print(tiny)
        tiny.load_state_dict(model.state_dict())
        test_two_modules(model, tiny, [2, 3, 224, 224])

    test_attn()
    test_model()
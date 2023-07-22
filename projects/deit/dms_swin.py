from mmengine.model import BaseModel
from mmrazor.models.mutables.derived_mutable import DerivedMutable
import torch
from mmrazor.implementations.pruning.dms.core.algorithm import BaseDTPAlgorithm, BaseAlgorithm, DmsAlgorithmMixin, update_dict_reverse
from mmrazor.implementations.pruning.dms.core.op import ImpLinear
from mmrazor.implementations.pruning.dms.core.dtp import DTPAMutator
from mmrazor.implementations.pruning.dms.core.models.opt.opt_analyzer import OutChannel, InChannel
from mmrazor.models.mutables import DerivedMutable, SequentialMutableChannelUnit

from mmcls.models.backbones.swin_transformer import SwinBlockSequence, SwinBlock, SwinTransformer, resize_pos_embed
from mmcls.models.utils.attention import WindowMSA
from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from mmcls.registry import MODELS as CLSMODELS
from mmrazor.registry import MODELS, TASK_UTILS
import torch.nn as nn
import copy
from mmcv.cnn.bricks.wrappers import Linear as MMCVLinear
from mmrazor.implementations.pruning.dms.core.op import (ImpModuleMixin,
                                                         DynamicBlockMixin,
                                                         MutableAttn,
                                                         QuickFlopMixin,
                                                         ImpLinear)
from mmcv.cnn.bricks.drop import DropPath
from mmrazor.implementations.pruning.dms.core.mutable import (
    ImpMutableChannelContainer, MutableChannelForHead, MutableChannelWithHead,
    MutableHead)
import torch.utils.checkpoint as cp
from dms_deit import MMCVLinear
from mmrazor.implementations.pruning.dms.core.unit import DTPTUnit
from mmrazor.implementations.pruning.dms.core.mutable import DTPTMutableChannelImp, DrivedDTPMutableChannelImp
from mmrazor.utils import print_log
from mmrazor.models.task_modules.estimators.counters import BaseCounter


# ops ####################################################################################
@TASK_UTILS.register_module()
class WindowMSACounter(BaseCounter):

    @staticmethod
    def add_count_hook(module: WindowMSA, input, output):
        """Calculate FLOPs and params based on the size of input & output."""
        # Can have multiple inputs, getting the first one
        head = module.num_heads
        B, N, C = input[0].shape

        qk_dim = module.qkv.out_features // 3 // head
        v_dim = module.qkv.out_features // 3 // head

        flops = 0

        print(B,head,N,qk_dim,v_dim)
        flops += B * head * N * N * (qk_dim + v_dim)  # attn

        module.__flops__ += flops


@CLSMODELS.register_module()
class SwinTransformer2(SwinTransformer):
    arch_zoo = {
    **dict.fromkeys(['t', 'tiny'],
                    {'embed_dims': 128,
                     'depths':     [4, 4,  12,  4],
                     'num_heads':  [4, 8, 16, 32]}),
    **dict.fromkeys(['s', 'small'],
                    {'embed_dims': 96,
                     'depths':     [2, 2, 18,  2],
                     'num_heads':  [3, 6, 12, 24]}),
    **dict.fromkeys(['b', 'base'],
                    {'embed_dims': 128,
                     'depths':     [2, 2, 18,  2],
                     'num_heads':  [4, 8, 16, 32]}),
    **dict.fromkeys(['l', 'large'],
                    {'embed_dims': 192,
                     'depths':     [2,  2, 18,  2],
                     'num_heads':  [6, 12, 24, 48]}),
    }  # yapf: disable

def swin_forward(self: SwinTransformer, x):
    x, hw_shape = self.patch_embed(x)
    if self.use_abs_pos_embed:
        x = x + resize_pos_embed(self.absolute_pos_embed,
                                 self.patch_resolution, hw_shape,
                                 self.interpolate_mode, self.num_extra_tokens)
    x = self.drop_after_pos(x)

    outs = []
    for i, stage in enumerate(self.stages):
        x, hw_shape = stage(
            x, hw_shape, do_downsample=self.out_after_downsample)
        if i in self.out_indices:
            norm_layer = getattr(self, f'norm{i}')
            out = norm_layer(x)
            out = out.view(out.shape[0], *hw_shape,
                           -1).permute(0, 3, 1, 2).contiguous()
            outs.append(out)
        if stage.downsample is not None and not self.out_after_downsample:
            x, hw_shape = stage.downsample(x, hw_shape)

    return tuple(outs)


SwinTransformer.forward = swin_forward


class DySwinBlock(SwinBlock, DynamicBlockMixin):

    def __init__(self, *args, **kwargs):
        SwinBlock.__init__(self, *args, **kwargs)
        DynamicBlockMixin.__init__(self)
        self.init_args = args
        self.init_kwargs = kwargs

    @classmethod
    def convert_from(cls, module: SwinBlock):
        new_module = cls(
            embed_dims=100,
            num_heads=2,
            window_size=7,
            shift=False,
            ffn_ratio=4.,
            drop_path=0.,
            pad_small_map=False,
            attn_cfgs=dict(),
            ffn_cfgs=dict(),
            norm_cfg=dict(type='LN'),
            with_cp=False,
            init_cfg=None)

        new_module.with_cp = module.with_cp

        for name, child in module.named_children():
            setattr(new_module, name, child)
        new_module.load_state_dict(module.state_dict(), strict=False)
        return new_module

    def to_static_op(self):
        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static

        module = self.static_op_factory(*self.init_args, **self.init_kwargs)

        module.with_cp = self.with_cp

        for name, m in self.named_children():  # type: ignore
            assert hasattr(module, name)
            setattr(module, name, _dynamic_to_static(m))
        module.ffn.add_identity = True
        return module

    @property
    def static_op_factory(self):
        return DySwinBlock

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x * self.scale + identity

            identity = x
            x = self.norm2(x)
            self.ffn.add_identity = False
            x = identity + self.ffn(x, identity=identity) * self.scale

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinLayers(nn.Sequential):

    @classmethod
    def convert_from(cls, module):
        return cls(module._modules)


class SplitWindowMSA(WindowMSA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        qkv_bias = self.qkv.bias is not None
        self.q = nn.Linear(self.embed_dims, self.embed_dims, bias=qkv_bias)
        self.k = nn.Linear(self.embed_dims, self.embed_dims, bias=qkv_bias)
        self.v = nn.Linear(self.embed_dims, self.embed_dims, bias=qkv_bias)
        self.proj = nn.Linear(
            self.embed_dims, self.embed_dims, bias=self.proj.bias is not None)
        delattr(self, 'qkv')

        self.init_kargs = kwargs
        self.init_args = args

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B_, N, C = x.shape

        ##########################################################################################
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
        #                           C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[
        #     2]  # make torchscript happy (cannot use tensor as tuple)

        q, k, v = self.q(x), self.k(x), self.v(x)

        def reshape(x: torch.Tensor):
            return x.reshape([B_, N, self.num_heads, -1]).permute(0, 2, 1, 3)

        q, k, v = map(reshape, [q, k, v])

        ##########################################################################################
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @classmethod
    def convert_from(cls, attn: WindowMSA):
        module = cls(
            embed_dims=attn.embed_dims,
            window_size=attn.window_size,
            num_heads=attn.num_heads,
            qkv_bias=attn.qkv.bias is not None,
            qk_scale=attn.scale,
            attn_drop=attn.attn_drop.p,
            proj_drop=attn.proj_drop.p)
        module.load_state_dict(attn.state_dict(), strict=False)
        return module


class DySplitWindowMSA(SplitWindowMSA, DynamicChannelMixin, MutableAttn,
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
    def convert_from(cls, module: SplitWindowMSA):
        if not isinstance(module, SplitWindowMSA):
            module = SplitWindowMSA.convert_from(module)
        new_module = cls(*module.init_args, **module.init_kargs)
        new_module.load_state_dict(module.state_dict(), strict=False)
        return new_module

    @property
    def static_op_factory(self):
        return SplitWindowMSA

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
        self.q.use_out_imp = True
        self.k.use_out_imp = True
        return m_head, m_qk, m_v

    def to_static_op(self):

        if 'head' in self.attn_mutables:
            num_heads = int(self.attn_mutables['head'].mask.sum().item())
        else:
            num_heads = self.num_heads

        module: SplitWindowMSA = SplitWindowMSA(*self.init_args,
                                                **self.init_kwargs)
        module.q = self.q.to_static_op()
        module.k = self.k.to_static_op()
        module.v = self.v.to_static_op()
        module.proj = self.proj.to_static_op()

        head_mask = self.attn_mutables['head'].mask.bool()

        module.relative_position_bias_table.data = self.relative_position_bias_table[
            ..., head_mask]

        module.num_heads = num_heads
        module.head_dims = module.q.out_features // num_heads
        module.embed_dims = module.v.out_features

        module.init_kargs.update({
            'embed_dims': module.q.out_features,
            'num_heads': num_heads,
            'input_dims': module.q.in_features,
        })

        module.scale = (self.q.out_features // num_heads)**-0.5

        return module

    def soft_flop(self):
        flops = 0
        flops = flops + QuickFlopMixin.get_flop(self.q)
        flops = flops + QuickFlopMixin.get_flop(self.k)
        flops = flops + QuickFlopMixin.get_flop(self.v)
        flops = flops + QuickFlopMixin.get_flop(self.proj)

        mutable_head: MutableHead = self.attn_mutables['head']
        mutable_qk: MutableChannelForHead = self.attn_mutables['qk']
        mutable_v: MutableChannelForHead = self.attn_mutables['v']
        head = mutable_head.current_imp_flop.sum()
        qk_dim = mutable_qk.current_imp_flop.sum() / head
        v_dim = mutable_v.current_imp_flop.sum() / head
        B, N, _ = self.quick_flop_recorded_in_shape[0]

        flops = flops + head * N * N * (qk_dim + v_dim)
        flops = flops * B
        return flops


class MyDropPath(DropPath):

    def extra_repr(self) -> str:
        return str(self.drop_prob)


# mutator ####################################################################################


class SwinMutableChannel(DTPTMutableChannelImp):

    def expand_mutable_channel(self, expand_ratio) -> DerivedMutable:

        def _expand_mask():
            mask = self.current_mask
            mask = torch.unsqueeze(
                mask, -2).expand([expand_ratio] + list(mask.shape)).flatten(-2)
            return mask

        return DrivedDTPMutableChannelImp(_expand_mask, _expand_mask,
                                          expand_ratio, [self])


@MODELS.register_module()
class SwinUnit(DTPTUnit):

    def __init__(self, num_channels: int, extra_mapping=...) -> None:
        super().__init__(num_channels, extra_mapping)
        self.mutable_channel = SwinMutableChannel(num_channels)


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
                InChannel(prefix + 'attn.w_msa.q', block.attn.w_msa.q))
            unit.add_input_related(
                InChannel(prefix + 'attn.w_msa.k', block.attn.w_msa.k))
            unit.add_input_related(
                InChannel(prefix + 'attn.w_msa.v', block.attn.w_msa.v))
            unit.add_input_related(InChannel(prefix + 'norm2', block.norm2))
            unit.add_input_related(
                InChannel(prefix + 'ffn.layers.0.0', block.ffn.layers[0][0]))

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
                assert isinstance(channel.module, nn.Module)


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
        backbone: SwinTransformer = model.backbone
        backbone.stages[0].blocks = SwinLayers.convert_from(
            backbone.stages[0].blocks)
        backbone.stages[1].blocks = SwinLayers.convert_from(
            backbone.stages[1].blocks)
        backbone.stages[2].blocks = SwinLayers.convert_from(
            backbone.stages[2].blocks)
        backbone.stages[3].blocks = SwinLayers.convert_from(
            backbone.stages[3].blocks)

        default_mutator_kwargs = dict(
            prune_qkv=False,
            prune_block=True,
            dtp_mutator_cfg=dict(
                type='SwinMutator',
                channel_unit_cfg=dict(
                    type='SwinUnit',
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
                SplitWindowMSA: DySplitWindowMSA,
                WindowMSA: DySplitWindowMSA,
            },
            block_initilizer_kwargs=dict(
                stage_mixin_layers=[SwinLayers],
                dynamic_block_mapping={SwinBlock: DySwinBlock}),
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

    def to_static_model(self, reset=True, drop_path=0.2):
        model = super().to_static_model()
        backbone: SwinTransformer2 = model.backbone

        if drop_path != -1:
            num_blocks = sum([len(stage.blocks) for stage in backbone.stages])
            i = 0
            for stage in backbone.stages:
                for block in stage.blocks:
                    block: SwinBlock
                    drop_path_rate = drop_path * i / num_blocks
                    block.attn.drop = MyDropPath(
                        drop_path_rate
                    ) if drop_path_rate != 0 else nn.Identity()
                    block.ffn.dropout_layer = MyDropPath(
                        drop_path_rate
                    ) if drop_path_rate != 0 else nn.Identity()

                    i += 1
            assert i == num_blocks
        print_log(model)
        return model


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

    def rand_mask(mask):
        while True:
            mask = (torch.rand_like(mask) < 0.5).float()
            if mask.sum() != 0:
                break
        return mask

    alg.mutator.init_random_tayler()
    alg.mutator.scale_flop_to(model, alg.scheduler.init_flop * 0.8)
    print(alg.mutator.info())
    model = alg.to_static_model(drop_path=0.2)
    x = torch.rand([2, 3, 224, 224])
    model(x)

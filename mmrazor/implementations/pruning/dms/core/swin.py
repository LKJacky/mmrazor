# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.utils.attention import WindowMSA
from torch import Tensor
from torch.nn import Module
from torchvision.models.swin_transformer import (ShiftedWindowAttention,
                                                 SwinTransformer,
                                                 SwinTransformerBlock)

from mmrazor.implementations.pruning.dtp.modules.ops import QuickFlopMixin
from mmrazor.models.architectures.dynamic_ops import (DynamicChannelMixin,
                                                      DynamicLinear)
from mmrazor.models.architectures.ops.swin import BaseShiftedWindowAttention
from mmrazor.models.mutables import BaseMutable
from mmrazor.models.task_modules.estimators.counters import BaseCounter
from mmrazor.registry import MODELS, TASK_UTILS
from ...dtp.modules.mutable_channels import ImpMutableChannelContainer
from ...dtp.modules.ops import ImpLinear, soft_mask_sum
from .mutable import MutableChannelForHead, MutableChannelWithHead, MutableHead
from .op import DynamicBlockMixin


class DynamicShiftedWindowAttention(BaseShiftedWindowAttention,
                                    DynamicChannelMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs

        self.q = DynamicLinear.convert_from(self.q)
        self.k = DynamicLinear.convert_from(self.k)
        self.v = DynamicLinear.convert_from(self.v)
        self.proj = DynamicLinear.convert_from(self.proj)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

        self.in_channels = self.q.in_features
        self.out_channels = self.proj.out_features

        self.init_mutable()

    def register_mutable_attr(self, attr: str, mutable):
        if attr == 'in_channels':
            self.q.register_mutable_attr(attr, mutable)
            self.k.register_mutable_attr(attr, mutable)
            self.v.register_mutable_attr(attr, mutable)
        elif attr == 'out_channels':
            self.proj.register_mutable_attr(attr, mutable)
        else:
            raise NotImplementedError()
        self.mutable_attrs[attr] = mutable

    @classmethod
    def convert_from(cls, module: ShiftedWindowAttention):
        new_module = cls(
            dim=module.q.in_features,
            window_size=module.window_size,
            shift_size=module.shift_size,
            num_heads=module.num_heads,
            qkv_bias=module.q.bias is not None,
            proj_bias=module.proj.bias is not None,
            attention_dropout=module.attention_dropout,
            dropout=module.dropout,
        )
        new_module.load_state_dict(module.state_dict(), strict=False)
        return new_module

    def to_static_op(self) -> Module:
        module: WindowMSA = self.static_op_factory(*self.init_args,
                                                   **self.init_kwargs)
        module.num_heads = self.attn_mutables['head'].activated_channels

        head_mask = self.attn_mutables['head'].mask.bool()
        module.relative_position_bias_table = nn.Parameter(
            self.relative_position_bias_table[:, head_mask],
            requires_grad=True)

        # self.relative_position_bias_table=
        module.q = self.q.to_static_op()
        module.k = self.k.to_static_op()
        module.v = self.v.to_static_op()
        module.proj = self.proj.to_static_op()

        return module

    @property
    def static_op_factory(self):
        return BaseShiftedWindowAttention

    def init_mutable(self):
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

        delattr(self, 'qkv')

        ImpMutableChannelContainer.register_mutable_channel_to_module(
            self.proj, mutable_v, is_to_output_channel=False)

        self.attn_mutables = {'head': m_head, 'qk': m_qk, 'v': m_v}

        return m_head, m_qk, m_v

    def forward(self, x, mask=None):
        B, H, W, _ = x.shape
        window_size = self.window_size
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_H = pad_b + H
        pad_W = pad_r + W

        x = self.shift_x(x)

        B_, N, C = x.shape
        num_head = self.attn_mutables['head'].activated_channels
        qk_dim = self.attn_mutables[
            'qk'].activated_channels // self.attn_mutables['head'].num_heads
        v_dim = self.attn_mutables[
            'v'].activated_channels // self.attn_mutables['head'].num_heads

        q = self.q(x).reshape(B_, N, num_head, qk_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, num_head, qk_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B_, N, num_head, v_dim).permute(0, 2, 1, 3)

        q = q * (q.shape[-1]**-0.5)
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.get_relative_position_bias()
        head_mask = self.attn_mutables['head'].mask.bool()
        relative_position_bias = relative_position_bias[:, head_mask, :, :]

        attn = attn + relative_position_bias

        mask = self.get_atten_mask(x, pad_H, pad_W)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, num_head, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, num_head, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = F.dropout(attn, p=self.attention_dropout)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = F.dropout(x, p=self.dropout)

        x = self.un_shift_x(x, B, H, W, pad_H, pad_W)
        return x

    def soft_flop(self):
        win_size = self.window_size[0] * self.window_size[1]
        _, H, W, _ = self.quick_flop_recorded_in_shape[0]
        n_win = H // self.window_size[0] * W // self.window_size[1]

        flops = 0
        flops = flops + self.q.soft_flop()
        flops = flops + self.k.soft_flop()
        flops = flops + self.v.soft_flop()
        flops = flops + self.proj.soft_flop()
        flops = flops * n_win

        head_ratio = soft_mask_sum(self.attn_mutables['head'].current_imp_flop
                                   ) / self.attn_mutables['head'].num_heads
        qk_dim = soft_mask_sum(self.q.output_imp_flop) * head_ratio
        v_dim = soft_mask_sum(self.v.output_imp_flop) * head_ratio

        flops = flops + n_win * (win_size**2) * (qk_dim + v_dim)
        return flops


class ImpShiftedWindowAttention(DynamicShiftedWindowAttention, QuickFlopMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._quick_flop_init()

        self.q = ImpLinear.convert_from(self.q)
        self.k = ImpLinear.convert_from(self.k)
        self.v = ImpLinear.convert_from(self.v)
        self.qkv = None
        self.proj = ImpLinear.convert_from(self.proj)

    def forward(self, x, mask=None):
        return BaseShiftedWindowAttention.forward(self, x, mask=mask)


class SwinSequential(nn.Sequential):
    pass


class DynamicSwinTransformerBlock(SwinTransformerBlock, DynamicBlockMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, attn_layer=BaseShiftedWindowAttention, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs
        self._dynamic_block_init()

    def forward(self, x):
        x = x + self.stochastic_depth(self.attn(self.norm1(x))) * self.scale
        x = x + self.stochastic_depth(self.mlp(self.norm2(x))) * self.scale
        return x

    def to_static_op(self) -> Module:

        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
        module = SwinTransformerBlock(*self.init_args, **self.init_kwargs)
        for name, m in self.named_children():
            assert hasattr(module, name)
            setattr(module, name, _dynamic_to_static(m))
        return module


@torch.fx.wrap
def _patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :].unsqueeze(-1)  # ... H/2 W/2 C 1
    x1 = x[..., 1::2, 0::2, :].unsqueeze(-1)  # ... H/2 W/2 C
    x2 = x[..., 0::2, 1::2, :].unsqueeze(-1)  # ... H/2 W/2 C
    x3 = x[..., 1::2, 1::2, :].unsqueeze(-1)  # ... H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1).flatten(-2, -1)  # ... H/2 W/2 C*4
    return x


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self,
                 dim: int,
                 norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        return x


@MODELS.register_module()
class TorchSwinBackbone(SwinTransformer):

    def __init__(self,
                 patch_size: List[int],
                 embed_dim: int,
                 depths: List[int],
                 num_heads: List[int],
                 window_size: List[int],
                 mlp_ratio: float = 4,
                 dropout: float = 0,
                 attention_dropout: float = 0,
                 stochastic_depth_prob: float = 0.1,
                 num_classes: int = 1000,
                 norm_layer=None,
                 block=DynamicSwinTransformerBlock,
                 downsample_layer=PatchMerging):
        super().__init__(patch_size, embed_dim, depths, num_heads, window_size,
                         mlp_ratio, dropout, attention_dropout,
                         stochastic_depth_prob, num_classes, norm_layer, block,
                         downsample_layer)
        delattr(self, 'avgpool')
        delattr(self, 'flatten')
        delattr(self, 'head')

        self.features[1] = SwinSequential(
            *(self.features[1]._modules.values()))
        self.features[3] = SwinSequential(
            *(self.features[3]._modules.values()))
        self.features[5] = SwinSequential(
            *(self.features[5]._modules.values()))
        self.features[7] = SwinSequential(
            *(self.features[7]._modules.values()))

        self.stochastic_depth_prob = stochastic_depth_prob

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        return (x, )

    def reset_sd_prob(self):
        blocks = list(self.features[1]._modules.values()) + list(
            self.features[3]._modules.values()) + list(
                self.features[5]._modules.values()) + list(
                    self.features[7]._modules.values())
        total_depth = len(blocks)
        from mmrazor.utils import print_log
        print_log(f'reset static depth prob with {total_depth} blocks')

        for i, block in enumerate(blocks):
            block: DynamicSwinTransformerBlock
            block.stochastic_depth.p = i / (total_depth -
                                            1) * self.stochastic_depth_prob

    def to_static_post_process(self):
        self.reset_sd_prob()


##########################################################################
# counters


@TASK_UTILS.register_module()
class BaseShiftedWindowAttentionCounter(BaseCounter):

    @staticmethod
    def add_count_hook(module: BaseShiftedWindowAttention, input, output):
        input = input[0]
        B, C, H, W = input.shape
        nH = H // module.window_size[0]
        nW = W // module.window_size[1]

        n_win = nH * nW
        win_size = module.window_size[0] * module.window_size[1]
        overall_flops = n_win * (win_size**2) * C * 2

        module.__flops__ += overall_flops


##########################################################################


class DynamicWindowMSA(WindowMSA, DynamicChannelMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs

        self.qkv = DynamicLinear.convert_from(self.qkv)
        self.proj = DynamicLinear.convert_from(self.proj)

    def register_mutable_attr(self, attr: str, mutable):
        if attr == 'in_channels':
            self.qkv.register_mutable_attr(attr, mutable)
        elif attr == 'out_channels':
            self.proj.register_mutable_attr(attr, mutable)
        else:
            raise NotImplementedError()

    @classmethod
    def convert_from(cls, module: WindowMSA):
        return cls(
            embed_dims=module.embed_dims,
            window_size=module.window_size,
            num_heads=module.num_heads,
            qkv_bias=module.qkv.bias is not None,
            qk_scale=module.scale,
            attn_drop=module.attn_drop.p,
            proj_drop=module.proj_drop.p,
            init_cfg=None)

    def to_static_op(self) -> Module:
        module: WindowMSA = self.static_op_factory(*self.init_args,
                                                   **self.init_kwargs)
        module.qkv = self.qkv
        module.proj = self.proj
        return module

    @property
    def static_op_factory(self):
        return WindowMSA

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  self.qkv.out_features // 3 //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

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

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.utils.attention import WindowMSA
from torch.nn import Module
from torchvision.models.swin_transformer import (PatchMerging,
                                                 ShiftedWindowAttention,
                                                 SwinTransformer,
                                                 SwinTransformerBlock)

from mmrazor.models.architectures.dynamic_ops import (DynamicChannelMixin,
                                                      DynamicLinear)
from mmrazor.models.mutables import BaseMutable
from mmrazor.registry import MODELS
from ...dtp.modules.ops import ImpLinear
from .op import DynamicBlockMixin


class DynamicShiftedWindowAttention(ShiftedWindowAttention,
                                    DynamicChannelMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs

        self.qkv = DynamicLinear.convert_from(self.qkv)
        self.proj = DynamicLinear.convert_from(self.proj)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

        self.in_channels = self.qkv.in_features
        self.out_channels = self.proj.out_features

    def register_mutable_attr(self, attr: str, mutable):
        if attr == 'in_channels':
            self.qkv.register_mutable_attr(attr, mutable)
        elif attr == 'out_channels':
            self.proj.register_mutable_attr(attr, mutable)
        else:
            raise NotImplementedError()
        self.mutable_attrs[attr] = mutable

    @classmethod
    def convert_from(cls, module: ShiftedWindowAttention):
        new_module = cls(
            dim=module.qkv.in_features,
            window_size=module.window_size,
            shift_size=module.shift_size,
            num_heads=module.num_heads,
            qkv_bias=module.qkv.bias is not None,
            proj_bias=module.proj.bias is not None,
            attention_dropout=module.attention_dropout,
            dropout=module.dropout,
        )
        new_module.load_state_dict(module.state_dict())
        return new_module

    def to_static_op(self) -> Module:
        module: WindowMSA = self.static_op_factory(*self.init_args,
                                                   **self.init_kwargs)
        module.qkv = self.qkv
        module.proj = self.proj
        module.load_state_dict(self.state_dict)
        return module

    @property
    def static_op_factory(self):
        return ShiftedWindowAttention

    def shift_x(self, input: torch.Tensor):
        B, H, W, C = input.shape
        window_size = self.window_size
        shift_size = self.shift_size
        # pad feature maps to multiples of window size
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x.shape

        shift_size = shift_size.copy()
        # If window size is larger than feature size, there is no need to shift window # noqa
        if window_size[0] >= pad_H:
            shift_size[0] = 0
        if window_size[1] >= pad_W:
            shift_size[1] = 0

        # cyclic shift
        if sum(shift_size) > 0:
            x = torch.roll(
                x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

        # partition windows
        num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
        x = x.view(B, pad_H // window_size[0], window_size[0],
                   pad_W // window_size[1], window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4,
                      5).reshape(B * num_windows,
                                 window_size[0] * window_size[1],
                                 C)  # B*nW, Ws*Ws, C
        return x

    def un_shift_x(self, x, B, H, W, pad_H, pad_W):
        window_size = self.window_size
        shift_size = self.shift_size
        C = x.shape[-1]
        # reverse windows
        x = x.view(B, pad_H // window_size[0], pad_W // window_size[1],
                   window_size[0], window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        # reverse cyclic shift
        if sum(shift_size) > 0:
            x = torch.roll(
                x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, :H, :W, :].contiguous()
        return x

    def get_atten_mask(self, x, pad_H, pad_W):
        shift_size = self.shift_size
        window_size = self.window_size
        pad_H, pad_W = pad_H, pad_W
        num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])

        if sum(shift_size) > 0:
            # generate attention mask
            attn_mask = x.new_zeros((pad_H, pad_W))
            h_slices = ((0, -window_size[0]),
                        (-window_size[0], -shift_size[0]), (-shift_size[0],
                                                            None))
            w_slices = ((0, -window_size[1]),
                        (-window_size[1], -shift_size[1]), (-shift_size[1],
                                                            None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0]:h[1], w[0]:w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0],
                                       pad_W // window_size[1], window_size[1])
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
                num_windows, window_size[0] * window_size[1])
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
            return attn_mask
        else:
            return None

    def forward(self, x, mask=None):
        B, H, W, _ = x.shape
        window_size = self.window_size
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_H = pad_b + H
        pad_W = pad_r + W

        x = self.shift_x(x)

        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  self.qkv.out_features // 3 //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * (q.shape[-1]**-0.5)
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.get_relative_position_bias()
        attn = attn + relative_position_bias

        mask = self.get_atten_mask(x, pad_H, pad_W)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = F.dropout(attn, p=self.attention_dropout)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = F.dropout(x, p=self.dropout)

        x = self.un_shift_x(x, B, H, W, pad_H, pad_W)
        return x


class ImpShiftedWindowAttention(DynamicShiftedWindowAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.qkv = ImpLinear.convert_from(self.qkv)
        self.proj = ImpLinear.convert_from(self.proj)


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


class SwinSequential(nn.Sequential):
    pass


class DynamicSwinTransformerBlock(SwinTransformerBlock, DynamicBlockMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        return (x, )

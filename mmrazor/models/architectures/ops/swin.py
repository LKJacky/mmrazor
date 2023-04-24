# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torchvision.models.swin_transformer import ShiftedWindowAttention


class BaseShiftedWindowAttention(ShiftedWindowAttention):

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

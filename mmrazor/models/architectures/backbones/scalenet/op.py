# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import math
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

OPS = OrderedDict()
# CAUTION: The assign order is Strict

OPS['ir_3x3_nse'] = lambda inp, oup, t, stride, n, inp_base, oup_base: InvertedResidual(
    inp=inp,
    oup=oup,
    t=t,
    stride=stride,
    k=3,
    n=n,
    inp_base=inp_base,
    oup_base=oup_base,
    activation=HSwish,
    use_se=False)
OPS['ir_5x5_nse'] = lambda inp, oup, t, stride, n, inp_base, oup_base: InvertedResidual(
    inp=inp,
    oup=oup,
    t=t,
    stride=stride,
    k=5,
    n=n,
    inp_base=inp_base,
    oup_base=oup_base,
    activation=HSwish,
    use_se=False)
OPS['ir_7x7_nse'] = lambda inp, oup, t, stride, n, inp_base, oup_base: InvertedResidual(
    inp=inp,
    oup=oup,
    t=t,
    stride=stride,
    k=7,
    n=n,
    inp_base=inp_base,
    oup_base=oup_base,
    activation=HSwish,
    use_se=False)
OPS['ir_3x3_se'] = lambda inp, oup, t, stride, n, inp_base, oup_base: InvertedResidual(
    inp=inp,
    oup=oup,
    t=t,
    stride=stride,
    k=3,
    n=n,
    inp_base=inp_base,
    oup_base=oup_base,
    activation=HSwish,
    use_se=True)
OPS['ir_5x5_se'] = lambda inp, oup, t, stride, n, inp_base, oup_base: InvertedResidual(
    inp=inp,
    oup=oup,
    t=t,
    stride=stride,
    k=5,
    n=n,
    inp_base=inp_base,
    oup_base=oup_base,
    activation=HSwish,
    use_se=True)
OPS['ir_7x7_se'] = lambda inp, oup, t, stride, n, inp_base, oup_base: InvertedResidual(
    inp=inp,
    oup=oup,
    t=t,
    stride=stride,
    k=7,
    n=n,
    inp_base=inp_base,
    oup_base=oup_base,
    activation=HSwish,
    use_se=True)

OPS['id'] = lambda inp, oup, t, stride, n, inp_base, oup_base: Identity(
    inp=inp,
    oup=oup,
    t=t,
    stride=stride,
    k=1,
    n=n,
    inp_base=inp_base,
    oup_base=oup_base)
OPS['conv2d'] = lambda inp, oup, _, stride, n, inp_base, oup_base, if_conv_out=False: ScaleConv2d(
    inp=inp,
    oup=oup,
    stride=stride,
    k=1,
    n=n,
    inp_base=inp_base,
    oup_base=oup_base,
    activation=HSwish,
    if_conv_out=if_conv_out)
OPS['conv3x3'] = lambda inp, oup, _, stride, n, inp_base, oup_base, if_conv_out=False: ScaleConv2d(
    inp=inp,
    oup=oup,
    stride=stride,
    k=3,
    n=n,
    inp_base=inp_base,
    oup_base=oup_base,
    activation=HSwish,
    if_conv_out=if_conv_out)

channel_mults = [1.0, 0.8, 0.6, 0.4, 0.2]

# class DynamicLinear(nn.Linear):
#     def __init__(self, *args, **kwargs):
#         self.n = kwargs.pop('n', 1)
#         super().__init__(*args, **kwargs)
#         self.idx = 0 # index of n-laterally couplng parts

#     def forward(self, input):
#         in_dim = input.shape[-1]
#         if self.idx % 2 == 0:
#             w = self.weight[:, :in_dim].contiguous()
#         else:
#             w = self.weight[:, (self.in_features - in_dim):].contiguous()

#         if self.n > 0:
#                 self.idx = (self.idx + 1) % self.n
#         return F.linear(input, w, self.bias)


class FC(nn.Module):

    def __init__(self, dim_in, dim_out, use_bn, dp=0., act='nn.ReLU', n=1):
        super(FC, self).__init__()
        self.inp = dim_in
        self.oup = dim_out
        self.module = []
        self.module.append(nn.Linear(dim_in, dim_out))
        if use_bn:
            self.module.append(nn.BatchNorm1d(dim_out))
        if act is not None:
            self.module.append(eval(act)(inplace=True))
        if dp != 0:
            self.module.append(nn.Dropout(dp))
        self.module = nn.Sequential(*self.module)

    # def forward_(self, x):
    #     if x.dim() != 2:
    #         x = x.flatten(1)
    #     return self.module(x)

    # @torch.jit.unused
    # def call_checkpoint_forward(self, x):
    #     def closure(*x):
    #         return self.forward_(*x)
    #     return checkpoint(closure, x)

    # def forward(self, x, n_c):
    #     return self.call_checkpoint_forward(x)
    #     # return self.forward_(x)

    def forward(self, x, n_c):
        if x.dim() != 2:
            x = x.flatten(1)
        return self.module(x)


class BasicOp(nn.Module):

    def __init__(self, oup, **kwargs):
        super(BasicOp, self).__init__()
        self.oup = oup
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def get_output_channles(self):
        return self.oup


class ScaleConv2d(BasicOp):

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 k,
                 inp_base=None,
                 oup_base=None,
                 activation=nn.ReLU,
                 n=1,
                 **kwargs):
        self.if_conv_out = kwargs.pop('if_conv_out', False)
        super(ScaleConv2d, self).__init__(oup, **kwargs)
        self.inp = inp
        self.oup = oup
        self.inp_base = inp_base  #inp of base model
        self.oup_base = oup_base  #oup of base model
        self.stride = stride
        self.k = k
        self.conv = nn.Conv2d(
            inp, oup, kernel_size=k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.act = activation()

    # def forward_(self, x):
    #     x = self.conv(x)
    #     x = self.bn(x)
    #     return self.act(x)

    # @torch.jit.unused
    # def call_checkpoint_forward(self, x):
    #     def closure(*x):
    #         return self.forward_(*x)
    #     return checkpoint(closure, x)

    # def forward(self, x, c_m):
    #     if not self.if_conv_out:
    #         self.conv.out_c = int(math.ceil(self.oup_base * c_m))
    #     return self.call_checkpoint_forward(x)
    #     # return self.forward_(x)

    def forward(self, x):
        # if not self.if_conv_out:
        #     self.conv.out_c = int(math.ceil(self.oup_base * c_m))
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class HSwish(nn.Module):

    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class HSigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class SqueezeExcite(nn.Module):

    def __init__(self,
                 in_channel,
                 reduction=4,
                 squeeze_act=nn.ReLU(inplace=True),
                 excite_act=HSigmoid(inplace=True),
                 n=1):
        super(SqueezeExcite, self).__init__()
        self.reduction = reduction
        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel // reduction,
            kernel_size=1,
            bias=True,
        )
        self.squeeze_act = squeeze_act
        self.excite_conv = nn.Conv2d(
            in_channels=in_channel // reduction,
            out_channels=in_channel,
            kernel_size=1,
            bias=True,
        )
        self.excite_act = excite_act

    def forward(self, inputs):
        self.squeeze_conv.out_c = inputs.size(1) // self.reduction
        self.excite_conv.out_c = inputs.size(1)
        # x = self.global_pooling(inputs)
        x = inputs.view(inputs.size(0), inputs.size(1), -1).mean(-1).view(
            inputs.size(0), inputs.size(1), 1, 1)
        x = self.squeeze_conv(x)
        x = self.squeeze_act(x)
        x = self.excite_conv(x)
        # x = self.excite_act(x)
        # return inputs * x
        return inputs * self.excite_act(x)


class InvertedResidual(BasicOp):

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 t,
                 inp_base=None,
                 oup_base=None,
                 k=3,
                 activation=nn.ReLU,
                 use_se=False,
                 n=1,
                 **kwargs):
        super(InvertedResidual, self).__init__(oup, **kwargs)
        self.stride = stride
        self.t = t
        self.k = k
        self.use_se = use_se
        assert stride in [1, 2]
        self.inp = inp
        self.oup = oup
        self.inp_base = inp_base  #inp of base model
        self.oup_base = oup_base  #oup of base model
        hidden_dim = int(round(inp * t))
        self.hidden_dim = hidden_dim

        if t == 1:
            # dw
            self.conv1 = nn.Conv2d(
                hidden_dim,
                hidden_dim,
                k,
                stride,
                padding=k // 2,
                groups=hidden_dim,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.act = activation(inplace=True)
            # se
            self.se = SqueezeExcite(
                hidden_dim, n=n) if use_se else nn.Sequential()
            # pw-linear
            self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn2 = nn.BatchNorm2d(oup)
        else:
            # pw
            self.conv1 = nn.Conv2d(
                inp,
                hidden_dim,
                1,
                1,
                0,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.act1 = activation(inplace=True)
            # dw
            self.conv1_2 = nn.Conv2d(
                hidden_dim,
                hidden_dim,
                k,
                stride,
                padding=k // 2,
                groups=hidden_dim,
                bias=False,
            )
            self.bn1_2 = nn.BatchNorm2d(hidden_dim)
            self.act2 = activation(inplace=True)
            # se
            self.se = SqueezeExcite(
                hidden_dim, n=n) if use_se else nn.Sequential()
            # pw-linear
            self.conv2 = nn.Conv2d(
                hidden_dim,
                oup,
                1,
                1,
                0,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(oup)

        self.use_shortcut = inp == oup and stride == 1

    # def forward_(self, x):
    #     if self.t == 1:
    #         y = self.conv1(x)
    #         y = self.bn1(y)
    #         y = self.act(y)
    #         y = self.se(y)
    #         y = self.conv2(y)
    #         y = self.bn2(y)
    #     else:
    #         y = self.conv1(x)
    #         y = self.bn1(y)
    #         y = self.act1(y)
    #         y = self.conv1_2(y)
    #         y = self.bn1_2(y)
    #         y = self.act2(y)
    #         y = self.se(y)
    #         y = self.conv2(y)
    #         y = self.bn2(y)
    #     if self.use_shortcut:
    #         y += x
    #     return y

    # @torch.jit.unused
    # def call_checkpoint_forward(self, x):
    #     def closure(*x):
    #         return self.forward_(*x)
    #     return checkpoint(closure, x)

    # def forward(self, x, c_m):
    #     self.conv1.out_c = int(math.ceil(int(round(self.inp_base * self.t)) * c_m))
    #     self.conv2.out_c = int(math.ceil(self.oup_base * c_m))
    #     # if not self.use_shortcut:
    #     #     self.conv1_2.out_c = self.conv1.out_c
    #     return self.call_checkpoint_forward(x)
    #     # return self.forward_(x)

    def forward(self, x):
        # self.conv1.out_c = int(
        #     math.ceil(int(round(self.inp_base * self.t)) * c_m))
        # self.conv2.out_c = int(math.ceil(self.oup_base * c_m))
        # if not self.use_shortcut:
        #     self.conv1_2.out_c = self.conv1.out_c
        if self.t == 1:
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.act(y)
            y = self.se(y)
            y = self.conv2(y)
            y = self.bn2(y)
        else:
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.act1(y)
            y = self.conv1_2(y)
            y = self.bn1_2(y)
            y = self.act2(y)
            y = self.se(y)
            y = self.conv2(y)
            y = self.bn2(y)
        if self.use_shortcut:
            y += x
        return y


class Identity(BasicOp):

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 inp_base=None,
                 oup_base=None,
                 **kwargs):
        super(Identity, self).__init__(oup, **kwargs)
        n = kwargs.pop('n', 1)
        self.inp = inp
        self.oup = oup
        self.inp_base = inp_base  #inp of base model
        self.oup_base = oup_base  #oup of base model
        if stride != 1 or inp != oup:
            self.downsample = True
            self.conv = nn.Conv2d(
                inp, oup, kernel_size=1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(oup)
        else:
            self.downsample = False

    def forward(self, x, c_m):
        if self.downsample:
            self.conv.out_c = int(math.ceil(self.oup_base * c_m))
            x = self.conv(x)
            x = self.bn(x)
        return x


class AveragePooling(BasicOp):

    def __init__(self, oup, **kwargs):
        super(AveragePooling, self).__init__(oup, **kwargs)
        self.pool = nn.AdaptiveAvgPool2d(oup)

    # def forward_(self, x):
    #     return self.pool(x)

    # @torch.jit.unused
    # def call_checkpoint_forward(self, x):
    #     def closure(*x):
    #         return self.forward_(*x)
    #     return checkpoint(closure, x)

    # def forward(self, x, c_m):
    #     return self.call_checkpoint_forward(x)
    #     # return self.pool(x)

    def forward(self, x, c_m):
        return self.pool(x)

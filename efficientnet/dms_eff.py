from timm.models.efficientnet import efficientnet_b0, EfficientNet
from timm.models._efficientnet_blocks import InvertedResidual, DepthwiseSeparableConv
from timm.layers import BatchNormAct2d
from torch import nn, nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmrazor.implementations.pruning.dms.core.algorithm import DmsGeneralAlgorithm
from mmrazor.implementations.pruning.dms.core.op import DynamicBlockMixin
from mmrazor.models.architectures.dynamic_ops import (DynamicChannelMixin,
                                                      DynamicBatchNormMixin,
                                                      DynamicLinear)
import torch.nn as nn
import torch.nn.functional as F
# OPS ####################################################################################


class DynamicBatchNormAct2d(BatchNormAct2d, DynamicBatchNormMixin):

    def __init__(self,
                 num_features,
                 eps=0.00001,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 apply_act=True,
                 act_layer=nn.ReLU,
                 act_kwargs=None,
                 inplace=True,
                 drop_layer=None,
                 device=None,
                 dtype=None):
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats, apply_act, act_layer, act_kwargs,
                         inplace, drop_layer, device, dtype)
        self.mutable_attrs = nn.ModuleDict()

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return BatchNormAct2d

    def to_static_op(self: _BatchNorm) -> BatchNormAct2d:
        running_mean, running_var, weight, bias = self.get_dynamic_params()
        if 'num_features' in self.mutable_attrs:
            num_features = self.mutable_attrs['num_features'].current_mask.sum(
            ).item()
        else:
            num_features = self.num_features

        static_bn = BatchNormAct2d(
            num_features=num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        static_bn.drop = self.drop
        static_bn.act = self.act

        if running_mean is not None:
            static_bn.running_mean.copy_(running_mean)
            static_bn.running_mean = static_bn.running_mean.to(
                running_mean.device)
        if running_var is not None:
            static_bn.running_var.copy_(running_var)
            static_bn.running_var = static_bn.running_var.to(
                running_var.device)
        if weight is not None:
            static_bn.weight = nn.Parameter(weight)
        if bias is not None:
            static_bn.bias = nn.Parameter(bias)

        return static_bn

    def forward(self, x):
        running_mean, running_var, weight, bias = self.get_dynamic_params()

        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        # _assert(x.ndim == 4, f'expected 4D input (got {x.ndim}D input)')

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is
                                                           None)
        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean
            if not self.training or self.track_running_stats else None,
            running_var
            if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.act(x)
        return x

    @classmethod
    def convert_from(cls, module: BatchNormAct2d):
        new_module = DynamicBatchNormAct2d(
            num_features=module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        new_module.act = module.act
        new_module.drop = module.drop
        new_module.load_state_dict(module.state_dict())
        return new_module


class ImpBatchNormAct2d(DynamicBatchNormAct2d):

    def forward(self, x):
        y = BatchNormAct2d.forward(self, x)
        return y


# blocks ####################################################################################


class DynamicInvertedResidual(InvertedResidual, DynamicBlockMixin):

    def __init__(self, *args, **kwargs):
        InvertedResidual.__init__(self, *args, **kwargs)
        DynamicBlockMixin.__init__(self)
        self.init_args = args
        self.init_kwargs = kwargs

    @property
    def is_removable(self):
        return self.has_skip

    @classmethod
    def convert_from(cls, module: InvertedResidual):
        static = cls(
            in_chs=module.conv_pw.in_channels,
            out_chs=module.conv_pwl.out_channels,
            dw_kernel_size=module.conv_dw.kernel_size[0],
            stride=module.conv_dw.stride,
            dilation=module.conv_dw.dilation,
            pad_type=module.conv_dw.padding,
        )
        static.has_skip = module.has_skip
        static.conv_pw = module.conv_pw
        static.bn1 = module.bn1
        static.conv_dw = module.conv_dw
        static.bn2 = module.bn2
        static.se = module.se
        static.conv_pwl = module.conv_pwl
        static.bn3 = module.bn3
        static.drop_path = module.drop_path
        static.load_state_dict(module.state_dict())
        return static

    @property
    def static_op_factory(self):
        return InvertedResidual

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) * self.scale + shortcut
        return x

    def to_static_op(self):
        static = InvertedResidual(
            in_chs=module.conv_pw.in_channels,
            out_chs=module.conv_pwl.out_channels,
            dw_kernel_size=module.conv_dw.kernel_size[0],
            stride=module.conv_dw.stride,
            dilation=module.conv_dw.dilation,
            pad_type=module.conv_dw.padding,
        )
        static.has_skip = self.has_skip
        static.conv_pw = self.conv_pw
        static.bn1 = self.bn1
        static.conv_dw = self.conv_dw
        static.bn2 = self.bn2
        static.se = self.se
        static.conv_pwl = self.conv_pwl
        static.bn3 = self.bn3
        static.drop_path = self.drop_path
        static.load_state_dict(self.state_dict())
        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
        module = static
        for name, m in self.named_children():  # type: ignore
            assert hasattr(module, name)
            setattr(module, name, _dynamic_to_static(m))
        return module


class EffStage(nn.Sequential):

    @classmethod
    def convert_from(cls, module: nn.Sequential):
        return cls(module._modules)


# Algo ####################################################################################


class EffDmsAlgorithm(DmsGeneralAlgorithm):

    def __init__(self,
                 model: EfficientNet,
                 mutator_kwargs={},
                 scheduler_kargs={}) -> None:

        # nn.Sequential -> EffStage
        new_seq = nn.Sequential()
        for name, block in model.blocks.named_children():
            new_seq.add_module(name, EffStage.convert_from(block))
        model.blocks = new_seq
        print(model)
        default_mutator_kwargs = dict(
            prune_qkv=False,
            prune_block=True,
            dtp_mutator_cfg=dict(
                type='DTPAMutator',
                channel_unit_cfg=dict(
                    type='DTPTUnit',
                    default_args=dict(
                        extra_mapping={BatchNormAct2d: ImpBatchNormAct2d})),
                parse_cfg=dict(
                    _scope_='mmrazor',
                    type='ChannelAnalyzer',
                    demo_input=dict(
                        type='DefaultDemoInput',
                        input_shape=(1, 3, 224, 224),
                    ),
                    tracer_type='FxTracer',
                    extra_mapping={BatchNormAct2d: DynamicBatchNormAct2d},
                )),
            extra_module_mapping={},
            block_initilizer_kwargs=dict(
                stage_mixin_layers=[EffStage],
                dynamic_block_mapping={
                    InvertedResidual: DynamicInvertedResidual
                }),
        )
        default_scheduler_kargs = dict(
            flops_target=0.8,
            decay_ratio=0.8,
            refine_ratio=0.2,
            flop_loss_weight=1000,
            structure_log_interval=1000,
            by_epoch=True,
            target_scheduler='cos',
        )
        default_mutator_kwargs.update(mutator_kwargs)
        default_scheduler_kargs.update(scheduler_kargs)
        super().__init__(
            model,
            mutator_kwargs=default_mutator_kwargs,
            scheduler_kargs=default_scheduler_kargs,
        )


if __name__ == '__main__':
    model = efficientnet_b0()
    print(model)

    algo = EffDmsAlgorithm(model)
    print(algo.mutator.info())

    config = algo.mutator.dtp_mutator.config_template(with_channels=True)
    print(config)
    import json
    print(json.dumps(config['channel_unit_cfg']['units'], indent=4))
    print(algo.mutator.info())

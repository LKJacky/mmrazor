from timm.models.efficientnet import efficientnet_b0
from timm.layers import BatchNormAct2d
from torch import nn, nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmrazor.implementations.pruning.dms.core.algorithm import DmsGeneralAlgorithm
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


if __name__ == '__main__':
    # set_log_level('debug')
    model = efficientnet_b0()
    print(model)

    algo = DmsGeneralAlgorithm(
        model,
        mutator_kwargs=dict(
            prune_qkv=False,
            prune_block=False,
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
                    extra_mapping={BatchNormAct2d: DynamicBatchNormAct2d}),
            ),
            extra_module_mapping={}),
    )
    print(algo.mutator.info())
    # print(algo)

    config = algo.mutator.dtp_mutator.config_template(with_channels=True)
    print(config)
    import json
    print(json.dumps(config['channel_unit_cfg']['units'], indent=4))
    print(algo)

# Copyright (c) OpenMMLab. All rights reserved.
#############################################################################
_base_ = '../../resnet56/resnet56_pretrain.py'
pretrained_path = 'work_dirs/pretrained/resnet56_pretrain.pth'

grad_clip = -1
flops_target = 0.3

target_flop_ratio = 0.3
flop_loss_weight = 1
input_shape = (1, 3, 32, 32)

log_by_epoch = True
log_interval = 1
##############################################################################

custom_imports = dict(imports=['projects'])

architecture = _base_.model

if hasattr(_base_, 'data_preprocessor'):
    architecture.update({'data_preprocessor': _base_.data_preprocessor})
    data_preprocessor = None

architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture['_scope_'] = _base_.default_scope

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='BaseDTPAlgorithm',
    architecture=architecture,
    mutator_cfg=dict(
        type='DTPMutator',
        channel_unit_cfg=dict(type='DTPUnit', default_args=dict()),
        parse_cfg=dict(
            _scope_='mmrazor',
            type='ChannelAnalyzer',
            demo_input=dict(
                type='DefaultDemoInput',
                input_shape=input_shape,
            ),
            tracer_type='FxTracer'),
    ),
    scheduler=dict(
        type='DTPScheduler',
        flops_target=0.5,
        decay_ratio=0.6,
        refine_ratio=0.2,
        flop_loss_weight=flop_loss_weight),
)

paramwise_cfg = dict(custom_keys={
    'mutable_channel': dict(decay_mult=0.0),
})
optim_wrapper = _base_.optim_wrapper
optim_wrapper.update({'paramwise_cfg': paramwise_cfg})

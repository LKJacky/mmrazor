# Copyright (c) OpenMMLab. All rights reserved.
#############################################################################
_base_ = '../../vgg/vgg_pretrain.py'
pretrained_path = 'work_dirs/pretrained/vgg_pretrained.pth'

decay_ratio = 0.6
refine_ratio = 0.4
target_flop_ratio = 0.30
flop_loss_weight = 100

log_interval = 196

input_shape = (1, 3, 32, 32)

epoch = 15
train_cfg = dict(by_epoch=True, max_epochs=epoch)

origin_lr = _base_.optim_wrapper.optimizer.lr
prune_lr = origin_lr * 0.1

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
        flops_target=target_flop_ratio,
        decay_ratio=decay_ratio,
        refine_ratio=refine_ratio,
        flop_loss_weight=flop_loss_weight,
        structure_log_interval=log_interval),
)

paramwise_cfg = dict(custom_keys={
    'mutable_channel': dict(decay_mult=0.0),
})
optim_wrapper = _base_.optim_wrapper
optim_wrapper.update({
    'paramwise_cfg': paramwise_cfg,
    'optimizer': {
        'lr': prune_lr
    },
})

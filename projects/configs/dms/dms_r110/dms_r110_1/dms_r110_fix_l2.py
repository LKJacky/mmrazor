# Copyright (c) OpenMMLab. All rights reserved.
#############################################################################
_base_ = '../../../resnet56/resnet101_pretrain.py'
pretrained_path = 'work_dirs/pretrained/resnet56_super_pretrain.pth'

decay_ratio = 0.6
refine_ratio = 0.4
target_flop_ratio = 0.17
flop_loss_weight = 100
by_epoch = True
target_scheduler = 'cos'
loss_type = 'l2'

log_interval = 196

input_shape = (1, 3, 32, 32)

epoch = 100
train_cfg = dict(by_epoch=True, max_epochs=epoch)

mutator_lr = _base_.optim_wrapper.optimizer.lr * 0.1

if hasattr(_base_, 'param_scheduler'):
    delattr(_base_, 'param_scheduler')

# param_scheduler = dict(
#     type='CosineAnnealingLR',
#     T_max=epoch,
#     by_epoch=True,
#     begin=0,
#     end=epoch,
#     _scope_='mmcls')

##############################################################################

custom_imports = dict(imports=['projects'])

architecture = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='mmrazor.ResNetCifarDMS', num_blocks=[18, 18, 18],
        num_classes=10),
    head=dict(
        type='mmcls.LinearClsHead',
        num_classes=10,
        in_channels=64,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
    ))

if hasattr(_base_, 'data_preprocessor'):
    architecture.update({'data_preprocessor': _base_.data_preprocessor})
    data_preprocessor = None

# architecture['init_cfg'] = dict(
#     type='Pretrained', checkpoint=pretrained_path)  # noqa
architecture['_scope_'] = _base_.default_scope

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='BaseDTPAlgorithm',
    architecture=architecture,
    mutator_cfg=dict(
        type='DMSMutator',
        dtp_mutator_cfg=dict(
            type='DTPAMutator',
            channel_unit_cfg=dict(type='DTPTUnit', default_args=dict()),
            parse_cfg=dict(
                _scope_='mmrazor',
                type='ChannelAnalyzer',
                demo_input=dict(
                    type='DefaultDemoInput',
                    input_shape=input_shape,
                ),
                tracer_type='FxTracer'),
        ),
    ),
    scheduler=dict(
        type='DMSScheduler',
        flops_target=target_flop_ratio,
        decay_ratio=decay_ratio,
        refine_ratio=refine_ratio,
        flop_loss_weight=flop_loss_weight,
        structure_log_interval=log_interval,
        by_epoch=by_epoch,
        target_scheduler=target_scheduler,
        loss_type=loss_type,
    ),
)

custom_hooks = getattr(_base_, 'custom_hooks', []) + [
    dict(
        type='mmrazor.PruningStructureHook',
        by_epoch=False,
        interval=log_interval),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=1,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=input_shape,
        ),
        early_stop=False,
        save_ckpt_thr=[],
    ),
]

paramwise_cfg = dict(custom_keys={
    'mutator': dict(decay_mult=0.0, lr=mutator_lr),
})
optim_wrapper = _base_.optim_wrapper
optim_wrapper.update({
    'paramwise_cfg': paramwise_cfg,
})

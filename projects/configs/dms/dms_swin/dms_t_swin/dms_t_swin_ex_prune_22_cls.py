# Copyright (c) OpenMMLab. All rights reserved.
#############################################################################

_base_ = '../swin_pretrain.py'
pretrained_path = './work_dirs/pretrained/converted_mobilnetv2.pth'  # noqa

decay_ratio = 0.6
refine_ratio = 0.4
target_flop_ratio = 0.23
flop_loss_weight = 100
by_epoch = True
target_scheduler = 'cos'

log_interval = 1000

input_shape = (1, 3, 224, 224)

epoch = 30
train_cfg = dict(by_epoch=True, max_epochs=epoch)

model_lr = _base_.optim_wrapper.optimizer.lr
original_lr = _base_.optim_wrapper.optimizer.lr
mutator_lr = original_lr * 0.1

if hasattr(_base_, 'param_scheduler'):
    delattr(_base_, 'param_scheduler')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        end=20,
        convert_to_iter_based=True,
        _scope_='mmcls'),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-05,
        by_epoch=True,
        begin=20,
        _scope_='mmcls')
]
find_unused_parameters = True

##############################################################################

custom_imports = dict(imports=['projects'])

architecture = _base_.model

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
    'optimizer': {
        'lr': model_lr
    }
})

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=5,
    ), )

# train_dataloader = dict(batch_size=16)
# val_dataloader = dict(batch_size=16)

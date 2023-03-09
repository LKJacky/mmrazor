# Copyright (c) OpenMMLab. All rights reserved.
#############################################################################
_base_ = '../../../resnet50/resnet50_pretrain_cos_smooth.py'
pretrained_path = 'work_dirs/pretrained/resnet50_cos_smooth_140.pth'

decay_ratio = 0.6
refine_ratio = 0.4
target_flop_ratio = 0.25
flop_loss_weight = 100

log_interval = 1000

input_shape = (1, 3, 224, 224)

epoch = 14
train_cfg = dict(by_epoch=True, max_epochs=epoch)

mutator_lr = 0.001
model_lr = 0.001
original_lr = _base_.optim_wrapper.optimizer.lr
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
    scheduler=dict(
        type='DTPAScheduler',
        flops_target=target_flop_ratio,
        decay_ratio=decay_ratio,
        refine_ratio=refine_ratio,
        flop_loss_weight=flop_loss_weight,
        structure_log_interval=log_interval),
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
    'mutable_channel': dict(decay_mult=0.0, lr=mutator_lr),
})
optim_wrapper = _base_.optim_wrapper
optim_wrapper.update({
    'paramwise_cfg': paramwise_cfg,
    'optimizer': {
        'lr': model_lr
    }
})

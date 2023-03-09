# Copyright (c) OpenMMLab. All rights reserved.
#############################################################################
_base_ = '../../../resnet56/resnet56_pretrain.py'
pretrained_path = 'work_dirs/pretrained/resnet56_pretrain.pth'

input_shape = (1, 3, 32, 32)

iter = 3
train_cfg = dict(_delete_=True, by_epoch=False, max_iters=iter, val_interval=1)
batch_size = 256
train_dataloader = dict(batch_size=batch_size, )

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1), )

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
    type='ChipAlgorithm',
    architecture=architecture,
    mutator=dict(
        type='ChipMutator',
        channel_unit_cfg=dict(type='ChipUnit', default_args=dict()),
        parse_cfg=dict(
            _scope_='mmrazor',
            type='ChannelAnalyzer',
            demo_input=dict(
                type='DefaultDemoInput',
                input_shape=input_shape,
            ),
            tracer_type='FxTracer'),
    ),
)

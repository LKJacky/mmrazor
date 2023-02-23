#############################################################################
_base_ = '../../../../../../models/resnet50/resnet50_pretrain_cos_smooth.py'
pretrained_path = 'work_dirs/pretrained/resnet50_cos_smooth_140.pth'  # noqa
# pretrained_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa

imp_type = 'dtp'
grad_clip = -1
prune_iter_ratio = 0.6
index_revert = False
grad_mode = 'cos'

target_flop_ratio = 0.5
flop_loss_weight = 50
input_shape = (1, 3, 224, 224)

log_by_epoch = False
log_interval = 1000

epoch = 20

# optimizer setting
if hasattr(_base_, 'param_scheduler'):
    delattr(_base_, 'param_scheduler')

train_cfg = dict(by_epoch=True, max_epochs=epoch, val_interval=1)

paramwise_cfg = dict(custom_keys={'mutable_channel': dict(decay_mult=0.0)})

optim_wrapper = _base_.optim_wrapper
optim_wrapper.update({
    'paramwise_cfg': paramwise_cfg,
    'optimizer': dict(lr=0.01),
})

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
    type='DTPAlgorithm',
    architecture=architecture,
    mutator_cfg=dict(
        type='ImpMutator',
        channel_unit_cfg=dict(
            type='ImpUnit',
            default_args=dict(
                imp_type=imp_type,
                grad_clip=grad_clip,
                index_revert=index_revert,
                grad_mode=grad_mode,
            )),
        parse_cfg=dict(
            _scope_='mmrazor',
            type='ChannelAnalyzer',
            demo_input=dict(
                type='DefaultDemoInput',
                input_shape=input_shape,
            ),
            tracer_type='FxTracer'),
    ),
    target_flop=target_flop_ratio,
    flop_loss_weight=flop_loss_weight,
    prune_iter_ratio=prune_iter_ratio)

custom_hooks = getattr(_base_, 'custom_hooks', []) + [
    dict(
        type='mmrazor.PruningStructureHook',
        by_epoch=log_by_epoch,
        interval=log_interval),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=-1,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=input_shape,
        ),
        early_stop=False,
        save_ckpt_thr=[],
        log_interval=log_interval,
        log_by_epoch=log_by_epoch),
]

model_wrapper_cfg = dict(
    type='mmrazor.DTPAlgorithmDDP',
    broadcast_buffers=False,
)

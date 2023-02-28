#############################################################################
_base_ = '../../../../../../../projects/models/vgg/configs/vgg_pretrain.py'
pretrained_path = './work_dirs/pretrained/vgg_pretrained.pth'

imp_type = 'dtp_a'
grad_clip = -1
prune_iter_ratio = 0.5
update_ratio = 0.6

target_flop_ratio = 0.3
flop_loss_weight = 100
mutator_lr = 1e-2
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
    type='DTPAlgorithm',
    architecture=architecture,
    mutator_cfg=dict(
        type='ImpMutator',
        channel_unit_cfg=dict(
            type='ImpUnit',
            default_args=dict(
                imp_type=imp_type,
                grad_clip=grad_clip,
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
    prune_iter_ratio=prune_iter_ratio,
    update_ratio=update_ratio,
)

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

paramwise_cfg = dict(
    custom_keys={
        '.e': dict(lr=mutator_lr, decay_mult=0.0),
        '.v': dict(lr=mutator_lr, decay_mult=0.0),
    })
optim_wrapper = _base_.optim_wrapper
optim_wrapper.update({'paramwise_cfg': paramwise_cfg})

# 93.7900

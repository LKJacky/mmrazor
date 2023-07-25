#############################################################################
import os

_base_ = ['mmcls::resnet/resnet50_8xb256-rsb-a1-600e_in1k.py']
custom_imports = dict(imports=['projects'])

pruned_path = f"./work_dirs/{os.environ.get('JOB_NAME','default')}/{os.environ.get('PTH_NAME','epoch_30')}.pth"  # noqa

##############################################################################

architecture = _base_.model
architecture['backbone'] = dict(
    type='mmrazor.ResNetDMS',
    depth=152,
    num_stages=4,
    out_indices=(3, ),
    style='pytorch',
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    drop_path_rate=0.05,
)
architecture['_scope_'] = _base_.default_scope

if hasattr(_base_, 'data_preprocessor'):
    architecture.update({'data_preprocessor': _base_.data_preprocessor})
    data_preprocessor = None

algo = dict(
    _scope_='mmrazor',
    type='ResDmsAlgo',
    architecture=architecture,
    mutator_cfg=dict(
        # type='DMSMutator',
        dtp_mutator_cfg=dict(
            type='DTPAMutator',
            channel_unit_cfg=dict(type='DTPTUnit', default_args=dict()),
            parse_cfg=dict(
                _scope_='mmrazor',
                type='ChannelAnalyzer',
                demo_input=dict(
                    type='DefaultDemoInput',
                    input_shape=(1, 3, 224, 224),
                ),
                tracer_type='FxTracer'),
        ), ),
    scheduler=dict(
        # type='DMSScheduler',
        flops_target=1.0,
        decay_ratio=0.2,
        refine_ratio=0.2,
        flop_loss_weight=100,
        structure_log_interval=100,
        by_epoch=True,
        target_scheduler='cosine',
    ),
    init_cfg=dict(type='Pretrained', checkpoint=pruned_path),
)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='ResNetDmsSubModel',
    algorithm=algo,
    drop_path_rate=0.05)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=5,
    ), )

# train_dataloader = dict(batch_size=16)
# val_dataloader = dict(batch_size=16)

#############################################################################
import os

_base_ = './dms_r56super_prune.py'

pruned_path = f"./work_dirs/{os.environ['JOB_NAME']}/{os.environ.get('PTH_NAME','epoch_30')}.pth"  # noqa

epoch = 300
train_cfg = dict(by_epoch=True, max_epochs=epoch)

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[150, 225], gamma=0.1)

##############################################################################

architecture = dict(
    type='mmrazor.ResNetCifarSuper',
    ratio=2.0,
    single_layer_ratio=2.0,
    num_blocks=[6, 6, 12],
)

algorithm = dict(
    _scope_='mmrazor',
    type='mmrazor.BaseDTPAlgorithm',
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
                    input_shape=_base_.input_shape,
                ),
                tracer_type='FxTracer'),
        ),
    ),
    scheduler=dict(type='DMSScheduler', ),
)
algorithm['init_cfg'] = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherSubModel',
    algorithm=algorithm,
)
custom_hooks = _base_.custom_hooks[:-2]

custom_hooks.append(
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=1000,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=_base_.input_shape,
        ),
        early_stop=False,
        save_ckpt_thr=[],
    ), )

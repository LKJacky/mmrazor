#############################################################################
import os

_base_ = ['./dms_t_swin_ex_prune_22_cls.py']
pruned_path = f"./work_dirs/{os.environ['JOB_NAME']}/{os.environ.get('PTH_NAME','epoch_30')}.pth"  # noqa

epoch = 300
train_cfg = dict(by_epoch=True, max_epochs=epoch)

# restore lr
optim_wrapper = {'optimizer': {'lr': _base_.original_lr}}

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

algorithm = _base_.model
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

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
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=5,
    ), )

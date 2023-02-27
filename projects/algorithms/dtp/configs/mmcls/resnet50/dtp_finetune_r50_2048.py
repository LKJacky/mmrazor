#############################################################################
import os

_base_ = './dtp_prune_r50.py'
pruned_path = f"./work_dirs/{os.environ['JOB_NAME']}/{os.environ.get('PTH_NAME','epoch_20')}.pth"  # noqa

epoch = 140
# param_scheduler = dict(
#     type='CosineAnnealingLR',
#     T_max=epoch,
#     by_epoch=True,
#     begin=0,
#     end=epoch,
#     _scope_='mmcls')
train_cfg = dict(by_epoch=True, max_epochs=epoch, val_interval=1)

#  change batch size to 2048
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=True,
        begin=0,
        # about 2500 iterations for ImageNet-1k
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=epoch - 5,
        by_epoch=True,
        begin=5,
        end=epoch,
    )
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)

batch_size = 256
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=batch_size)
test_evaluator = dict(batch_size=batch_size)

##############################################################################

algorithm = _base_.model
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherSubModel',
    algorithm=algorithm,
)

# remove pruning related hooks
custom_hooks = _base_.custom_hooks[:-2]

# delete ddp
model_wrapper_cfg = None

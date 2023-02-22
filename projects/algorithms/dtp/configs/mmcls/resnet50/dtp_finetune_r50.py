#############################################################################
import os

_base_ = './dtp_prune_r50.py'
pruned_path = f"./work_dirs/{os.environ['JOB_NAME']}/epoch_20.pth"

epoch = 140
param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=epoch,
    by_epoch=True,
    begin=0,
    end=epoch,
    _scope_='mmcls')
train_cfg = dict(by_epoch=True, max_epochs=epoch, val_interval=1)
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

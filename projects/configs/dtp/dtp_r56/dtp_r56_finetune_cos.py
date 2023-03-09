#############################################################################
import os

_base_ = './dtp_r56_prune.py'

pruned_path = f"./work_dirs/{os.environ['JOB_NAME']}/{os.environ.get('PTH_NAME','epoch_30')}.pth"  # noqa

epoch = 300
train_cfg = dict(by_epoch=True, max_epochs=epoch)

param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR',
    T_max=epoch,
    by_epoch=True,
    begin=0,
    end=epoch,
    _scope_='mmcls')

##############################################################################

algorithm = _base_.model
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherSubModel',
    algorithm=algorithm,
)

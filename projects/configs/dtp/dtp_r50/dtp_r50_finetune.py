#############################################################################
import os

_base_ = './dtp_r50_prune.py'

pruned_path = f"./work_dirs/{os.environ['JOB_NAME']}/{os.environ.get('PTH_NAME','epoch_30')}.pth"  # noqa

epoch = 140
train_cfg = dict(by_epoch=True, max_epochs=epoch)
# restore lr
optim_wrapper = {'optimizer': {'lr': _base_.original_lr}}
##############################################################################

algorithm = _base_.model
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherSubModel',
    algorithm=algorithm,
)

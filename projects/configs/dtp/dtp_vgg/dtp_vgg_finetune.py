#############################################################################
import os

_base_ = './dtp_vgg_prune.py'

pruned_path = f"./work_dirs/{os.environ['JOB_NAME']}/{os.environ.get('PTH_NAME','epoch_30')}.pth"  # noqa

epoch = 150
train_cfg = dict(by_epoch=True, max_epochs=epoch)

optim_wrapper = _base_.optim_wrapper
optim_wrapper.update({
    'optimizer': {
        'lr': _base_.origin_lr,
    },
})

##############################################################################

algorithm = _base_.model
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherSubModel',
    algorithm=algorithm,
)

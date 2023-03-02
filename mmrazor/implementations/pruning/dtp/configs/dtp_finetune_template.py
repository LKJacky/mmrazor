# Copyright (c) OpenMMLab. All rights reserved.
#############################################################################
_base_ = ''
pruned_path = ''
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

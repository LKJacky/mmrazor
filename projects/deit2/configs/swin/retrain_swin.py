#############################################################################

_base_ = ['./prune_swin.py']
pruned_path = f"./work_dirs/prune_swin/epoch_30.pth"  # noqa

epoch = 300
train_cfg = dict(by_epoch=True, max_epochs=epoch)

# restore lr
optim_wrapper = {'optimizer': {'lr': _base_.original_lr}}

find_unused_parameters = False

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

# custom_hooks.append(
#     dict(
#         type='mmrazor.ResourceInfoHook',
#         interval=1000,
#         demo_input=dict(
#             type='mmrazor.DefaultDemoInput',
#             input_shape=_base_.input_shape,
#         ),
#         early_stop=False,
#         save_ckpt_thr=[],
#     ), )

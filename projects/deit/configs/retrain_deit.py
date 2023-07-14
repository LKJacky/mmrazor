#############################################################################
import os

_base_ = ['./prune_deit.py']
pruned_path = f"work_dirs/prune_deit/epoch_30.pth"  # noqa

epoch = 300
train_cfg = dict(by_epoch=True, max_epochs=epoch)

# restore lr
optim_wrapper = {'optimizer': {'lr': _base_.original_lr}}

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20)
]

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
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=5,
    ), )

# train_dataloader = _base_.train_dataloader
# val_dataloader = _base_.val_dataloader
# test_dataloader = _base_.test_dataloader

# train_dataloader.batch_size = 128
# val_dataloader.batch_size = 128

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/coco':
#         's3://openmmlab/datasets/detection/coco',
#         'data/coco':
#         's3://openmmlab/datasets/detection/coco',
#         './data/cityscapes':
#         's3://openmmlab/datasets/segmentation/cityscapes',
#         'data/cityscapes':
#         's3://openmmlab/datasets/segmentation/cityscapes',
#         './data/imagenet':
#         's3://openmmlab/datasets/classification/imagenet',
#         'data/imagenet':
#         's3://openmmlab/datasets/classification/imagenet'
#     }))
# train_dataloader['dataset']['pipeline'][0][
#     'file_client_args'] = file_client_args
# val_dataloader['dataset']['pipeline'][0]['file_client_args'] = file_client_args
# test_dataloader['dataset']['pipeline'][0][
#     'file_client_args'] = file_client_args

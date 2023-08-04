_base_ = 'mmcls::swin_transformer/swin-base_16xb64_in1k.py'
custom_imports = dict(imports=['dms_deit','dms_swin'])

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer2', arch='tiny', img_size=224,
        drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)
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

# train_dataloader['pin_memory']=True
# val_dataloader['pin_memory']=True

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

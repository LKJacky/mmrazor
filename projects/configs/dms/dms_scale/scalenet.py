_base_ = ['./rsb.py']

# model = dict(
#     _delete_=True,
#     backbone=dict(
#         norm_cfg=dict(type='SyncBN', requires_grad=True),
#         drop_path_rate=0.05,
#     ),
#     head=dict(loss=dict(use_sigmoid=True)),
#     train_cfg=dict(augments=[
#         dict(type='Mixup', alpha=0.1),
#         dict(type='CutMix', alpha=1.0)
#     ]))

model = dict(
    type='ImageClassifier',
    backbone=dict(
        _delete_=True,
        type='mmrazor.ScaleNet',
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(use_sigmoid=True)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1),
        dict(type='CutMix', alpha=1.0)
    ]),
    _scope_='mmcls')

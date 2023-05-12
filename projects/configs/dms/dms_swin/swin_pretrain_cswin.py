_base_ = 'mmcls::swin_transformer/swin-tiny_16xb64_in1k.py'
model = _base_.model

model = dict(
    type='ImageClassifier',
    backbone=dict(
        _delete_=True,
        type='mmrazor.TorchSwinBackbone',
        patch_size=[4, 4],
        embed_dim=64,
        depths=[1, 2, 21, 1],
        num_heads=[2, 4, 8, 16],
        window_size=[7, 7],
        stochastic_depth_prob=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
    _scope_='mmcls')

train_dataloader = dict(batch_size=128)
val_dataloader = dict(batch_size=128)

# train_dataloader = dict(batch_size=4)
# val_dataloader = dict(batch_size=4)

_base_ = [
    './resnet56.py', '../cifar10/cifar10_bs16.py',
    '../cifar10/cifar10_bs128_300.py', '../cifar10/default_runtime.py'
]
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='mmrazor.ResNetCifar', num_blocks=[18, 18, 18], num_classes=10),
    head=dict(
        type='mmcls.LinearClsHead',
        num_classes=10,
        in_channels=64,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
    ))

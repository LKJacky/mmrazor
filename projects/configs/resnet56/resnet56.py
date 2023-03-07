model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(type='mmrazor.ResNetCifar', num_classes=10),
    head=dict(
        type='mmcls.LinearClsHead',
        num_classes=10,
        in_channels=64,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
    ))

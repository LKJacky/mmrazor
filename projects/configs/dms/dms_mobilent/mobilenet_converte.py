_base_ = 'mmcls::mobilenet_v2/mobilenet-v2_8xb32_in1k.py'

model = dict(
    type='ImageClassifier',
    backbone=dict(type='mmrazor.DmsMobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

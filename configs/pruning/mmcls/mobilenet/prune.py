model = dict(
    _scope_='mmrazor',
    type='ItePruneAlgorithm',
    architecture=dict(
        type='ImageClassifier',
        backbone=dict(type='MobileNetV2', widen_factor=1.0),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=1280,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5)),
        _scope_='mmcls',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa
            'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'  # noqa
        ),
        data_preprocessor=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)),
    target_pruning_ratio=dict({
        'backbone.conv1.conv_(0, 32)_32': 1.0,
        'backbone.layer1.0.conv.1.conv_(0, 16)_16': 1.0,
        'backbone.layer2.0.conv.0.conv_(0, 96)_96': 1.0,
        'backbone.layer2.0.conv.2.conv_(0, 24)_24': 1.0,
        'backbone.layer2.1.conv.0.conv_(0, 144)_144': 1.0,
        'backbone.layer3.0.conv.0.conv_(0, 144)_144': 1.0,
        'backbone.layer3.0.conv.2.conv_(0, 32)_32': 1.0,
        'backbone.layer3.1.conv.0.conv_(0, 192)_192': 1.0,
        'backbone.layer3.2.conv.0.conv_(0, 192)_192': 1.0,
        'backbone.layer4.0.conv.0.conv_(0, 192)_192': 1.0,
        'backbone.layer4.0.conv.2.conv_(0, 64)_64': 1.0,
        'backbone.layer4.1.conv.0.conv_(0, 384)_384': 1.0,
        'backbone.layer4.2.conv.0.conv_(0, 384)_384': 1.0,
        'backbone.layer4.3.conv.0.conv_(0, 384)_384': 1.0,
        'backbone.layer5.0.conv.0.conv_(0, 384)_384': 1.0,
        'backbone.layer5.0.conv.2.conv_(0, 96)_96': 1.0,
        'backbone.layer5.1.conv.0.conv_(0, 576)_576': 1.0,
        'backbone.layer5.2.conv.0.conv_(0, 576)_576': 1.0,
        'backbone.layer6.0.conv.0.conv_(0, 576)_576': 1.0,
        'backbone.layer6.0.conv.2.conv_(0, 160)_160': 1.0,
        'backbone.layer6.1.conv.0.conv_(0, 960)_960': 1.0,
        'backbone.layer6.2.conv.0.conv_(0, 960)_960': 1.0,
        'backbone.layer7.0.conv.0.conv_(0, 960)_960': 1.0,
        'backbone.layer7.0.conv.2.conv_(0, 320)_320': 1.0,
        'backbone.conv2.conv_(0, 1280)_1280': 1.0
    }),
    mutator_cfg=dict(
        type='ChannelMutator',
        channel_unit_cfg=dict(
            type='L1MutableChannelUnit',
            default_args=dict(choice_mode='ratio')),
        parse_cfg=dict(
            type='PruneTracer',
            tracer_type='FxTracer',
            demo_input=dict(type='DefaultDemoInput', scope='mmcls'))))
dataset_type = 'ImageNet'
data_preprocessor = None
train_pipeline = [
    dict(type='LoadImageFromFile', _scope_='mmcls'),
    dict(
        type='RandomResizedCrop', scale=224, backend='pillow',
        _scope_='mmcls'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal', _scope_='mmcls'),
    dict(type='PackClsInputs', _scope_='mmcls')
]
test_pipeline = [
    dict(type='LoadImageFromFile', _scope_='mmcls'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        _scope_='mmcls'),
    dict(type='CenterCrop', crop_size=224, _scope_='mmcls'),
    dict(type='PackClsInputs', _scope_='mmcls')
]
train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224, backend='pillow'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackClsInputs')
        ],
        _scope_='mmcls'),
    sampler=dict(type='DefaultSampler', shuffle=True, _scope_='mmcls'),
    persistent_workers=True)
val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ],
        _scope_='mmcls'),
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmcls'),
    persistent_workers=True)
val_evaluator = dict(type='Accuracy', topk=(1, 5), _scope_='mmcls')
test_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ],
        _scope_='mmcls'),
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmcls'),
    persistent_workers=True)
test_evaluator = dict(type='Accuracy', topk=(1, 5), _scope_='mmcls')
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.045,
        momentum=0.9,
        weight_decay=4e-05,
        _scope_='mmcls'))
param_scheduler = dict(
    type='StepLR', by_epoch=True, step_size=1, gamma=0.98, _scope_='mmcls')
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=256)
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmcls'),
    logger=dict(type='LoggerHook', interval=100, _scope_='mmcls'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmcls'),
    checkpoint=dict(type='CheckpointHook', interval=1, _scope_='mmcls'),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmcls'),
    visualization=dict(
        type='VisualizationHook', enable=False, _scope_='mmcls'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend', _scope_='mmcls')]
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    _scope_='mmcls')
log_level = 'INFO'
load_from = None
resume = False

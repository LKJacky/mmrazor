model = dict(
    _scope_='mmrazor',
    type='ItePruneAlgorithm',
    architecture=dict(
        type='ImageClassifier',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5)),
        _scope_='mmcls',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa
        ),
        data_preprocessor=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)),
    target_pruning_ratio=dict({
        'backbone.conv1_(0, 64)_64': 1.0,
        'backbone.layer1.0.conv1_(0, 64)_64': 1.0,
        'backbone.layer1.0.conv2_(0, 64)_64': 1.0,
        'backbone.layer1.0.conv3_(0, 256)_256': 1.0,
        'backbone.layer1.1.conv1_(0, 64)_64': 1.0,
        'backbone.layer1.1.conv2_(0, 64)_64': 1.0,
        'backbone.layer1.2.conv1_(0, 64)_64': 1.0,
        'backbone.layer1.2.conv2_(0, 64)_64': 1.0,
        'backbone.layer2.0.conv1_(0, 128)_128': 1.0,
        'backbone.layer2.0.conv2_(0, 128)_128': 1.0,
        'backbone.layer2.0.conv3_(0, 512)_512': 1.0,
        'backbone.layer2.1.conv1_(0, 128)_128': 1.0,
        'backbone.layer2.1.conv2_(0, 128)_128': 1.0,
        'backbone.layer2.2.conv1_(0, 128)_128': 1.0,
        'backbone.layer2.2.conv2_(0, 128)_128': 1.0,
        'backbone.layer2.3.conv1_(0, 128)_128': 1.0,
        'backbone.layer2.3.conv2_(0, 128)_128': 1.0,
        'backbone.layer3.0.conv1_(0, 256)_256': 1.0,
        'backbone.layer3.0.conv2_(0, 256)_256': 1.0,
        'backbone.layer3.0.conv3_(0, 1024)_1024': 1.0,
        'backbone.layer3.1.conv1_(0, 256)_256': 1.0,
        'backbone.layer3.1.conv2_(0, 256)_256': 1.0,
        'backbone.layer3.2.conv1_(0, 256)_256': 1.0,
        'backbone.layer3.2.conv2_(0, 256)_256': 1.0,
        'backbone.layer3.3.conv1_(0, 256)_256': 1.0,
        'backbone.layer3.3.conv2_(0, 256)_256': 1.0,
        'backbone.layer3.4.conv1_(0, 256)_256': 1.0,
        'backbone.layer3.4.conv2_(0, 256)_256': 1.0,
        'backbone.layer3.5.conv1_(0, 256)_256': 1.0,
        'backbone.layer3.5.conv2_(0, 256)_256': 1.0,
        'backbone.layer4.0.conv1_(0, 512)_512': 1.0,
        'backbone.layer4.0.conv2_(0, 512)_512': 1.0,
        'backbone.layer4.0.conv3_(0, 2048)_2048': 1.0,
        'backbone.layer4.1.conv1_(0, 512)_512': 1.0,
        'backbone.layer4.1.conv2_(0, 512)_512': 1.0,
        'backbone.layer4.2.conv1_(0, 512)_512': 1.0,
        'backbone.layer4.2.conv2_(0, 512)_512': 1.0
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
    dict(type='RandomResizedCrop', scale=224, _scope_='mmcls'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal', _scope_='mmcls'),
    dict(type='PackClsInputs', _scope_='mmcls')
]
test_pipeline = [
    dict(type='LoadImageFromFile', _scope_='mmcls'),
    dict(type='ResizeEdge', scale=256, edge='short', _scope_='mmcls'),
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
            dict(type='RandomResizedCrop', scale=224),
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
            dict(type='ResizeEdge', scale=256, edge='short'),
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
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ],
        _scope_='mmcls'),
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmcls'),
    persistent_workers=True)
test_evaluator = dict(type='Accuracy', topk=(1, 5), _scope_='mmcls')
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
        _scope_='mmcls'))
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[30, 60, 90],
    gamma=0.1,
    _scope_='mmcls')
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
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

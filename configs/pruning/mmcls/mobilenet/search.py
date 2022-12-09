model = dict(
    _scope_='mmrazor',
    type='SearchWrapper',
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
        data_preprocessor=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa
            'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'  # noqa
        )),
    mutator_cfg=dict(
        type='ChannelMutator',
        channel_unit_cfg=dict(
            type='L1MutableChannelUnit',
            default_args=dict(choice_mode='ratio')),
        parse_cfg=dict(
            type='PruneTracer',
            tracer_type='FxTracer',
            demo_input=dict(type='mmrazor.DefaultDemoInput', scope='mmcls'))))
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
train_cfg = dict(
    type='mmrazor.PruneEvolutionSearchLoop',
    dataloader=dict(
        batch_size=32,
        num_workers=5,
        dataset=dict(
            type='ImageNet',
            data_root='data/imagenet',
            ann_file='meta/val.txt',
            data_prefix='val',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='ResizeEdge',
                    scale=256,
                    edge='short',
                    backend='pillow'),
                dict(type='CenterCrop', crop_size=224),
                dict(type='PackClsInputs')
            ],
            _scope_='mmcls'),
        sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmcls'),
        persistent_workers=True),
    bn_dataloader=dict(
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
        persistent_workers=True),
    evaluator=dict(type='Accuracy', topk=(1, 5), _scope_='mmcls'),
    max_epochs=20,
    num_candidates=20,
    top_k=5,
    num_mutation=10,
    num_crossover=10,
    mutate_prob=0.2,
    flops_range=(0.49, 0.51),
    resource_estimator_cfg=dict(
        flops_params_cfg=dict(
            input_shape=(1, 3, 224, 224),
            input_constructor=dict(
                type='mmrazor.DefaultDemoInput', scope='mmcls'))),
    score_key='accuracy/top1')
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

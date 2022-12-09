model = dict(
    _scope_='mmrazor',
    type='SearchWrapper',
    architecture=dict(
        type='mmcls.ImageClassifier',
        backbone=dict(type='mmrazor.VGGCifar', num_classes=10),
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=10,
            in_channels=512,
            loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./work_dirs/pretrained/vgg_pretrained.pth'),
        data_preprocessor=None),
    mutator_cfg=dict(
        type='ChannelMutator',
        channel_unit_cfg=dict(
            type='L1MutableChannelUnit',
            default_args=dict(choice_mode='ratio')),
        parse_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(
                type='ImageClassifierPseudoLoss',
                input_shape=(2, 3, 32, 32)))))
dataset_type = 'CIFAR10'
preprocess_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs')
]
test_pipeline = [dict(type='PackClsInputs')]
train_dataloader = dict(
    batch_size=256,
    num_workers=2,
    dataset=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        test_mode=False,
        pipeline=[
            dict(type='RandomCrop', crop_size=32, padding=4),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True)
val_dataloader = dict(
    batch_size=256,
    num_workers=2,
    dataset=dict(
        type='mmcls.CIFAR10',
        data_prefix='data/cifar10/',
        test_mode=True,
        pipeline=[dict(type='PackClsInputs')]),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
val_evaluator = dict(type='mmcls.Accuracy', topk=(1, ))
test_dataloader = dict(
    batch_size=256,
    num_workers=2,
    dataset=dict(
        type='CIFAR10',
        data_prefix='data/cifar10/',
        test_mode=True,
        pipeline=[dict(type='PackClsInputs')]),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
test_evaluator = dict(type='Accuracy', topk=(1, ))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.005))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[50, 100], gamma=0.1)
train_cfg = dict(
    type='mmrazor.PruneEvolutionSearchLoop',
    dataloader=dict(
        batch_size=256,
        num_workers=2,
        dataset=dict(
            type='mmcls.CIFAR10',
            data_prefix='data/cifar10/',
            test_mode=True,
            pipeline=[dict(type='PackClsInputs')]),
        sampler=dict(type='DefaultSampler', shuffle=False),
        persistent_workers=True),
    evaluator=dict(type='mmcls.Accuracy', topk=(1, )),
    max_epochs=20,
    num_candidates=20,
    top_k=5,
    num_mutation=10,
    num_crossover=10,
    mutate_prob=0.2,
    flops_range=(0.45, 0.55),
    resource_estimator_cfg=dict(
        flops_params_cfg=dict(input_shape=(1, 3, 32, 32))),
    score_key='accuracy/top1')
val_cfg = dict()
test_cfg = dict()
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
data_preprocessor = None

model = dict(
    _scope_='mmrazor',
    type='ItePruneAlgorithm',
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
    target_pruning_ratio=dict({
        'backbone.features.conv0_(0, 64)_64':
        0.3547048912142781,
        'backbone.features.conv1_(0, 64)_64':
        0.9015415985029569,
        'backbone.features.conv3_(0, 128)_128':
        0.7727213528616353,
        'backbone.features.conv4_(0, 128)_128':
        0.8780924464336763,
        'backbone.features.conv6_(0, 256)_256':
        0.8605305975050029,
        'backbone.features.conv7_(0, 256)_256':
        0.6275234382979528,
        'backbone.features.conv8_(0, 256)_256':
        0.8987535637892307,
        'backbone.features.conv10_(0, 512)_512':
        0.875,
        'backbone.features.conv11_(0, 512)_512':
        0.6355129300922483,
        'backbone.features.conv12_(0, 512)_512':
        0.8509640262799506,
        'backbone.features.conv14_(0, 512)_512':
        0.32514615027975496,
        'backbone.features.conv15_(0, 512)_512':
        0.5422220856727952,
        'backbone.features.conv16_(0, 512)_512':
        0.8671875
    }),
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
        type='CIFAR10',
        data_prefix='data/cifar10/',
        test_mode=True,
        pipeline=[dict(type='PackClsInputs')]),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
val_evaluator = dict(type='Accuracy', topk=(1, ))
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
train_cfg = dict(by_epoch=True, max_epochs=150)
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

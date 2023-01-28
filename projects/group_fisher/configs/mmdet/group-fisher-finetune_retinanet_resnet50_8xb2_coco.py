_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'
custom_imports = dict(imports=['projects'])

architecture = _base_.model
architecture.backbone.frozen_stages = -1
# `pruned_path` need to be updated.
pruned_path = '/your/local/pruned/checkpoint/path'

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    pruning=False,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(type='GroupFisherChannelUnit')),
    init_cfg=dict(type='Pretrained', checkpoint=pruned_path),
)

optim_wrapper = dict(type='OptimWrapper', optimizer=dict(lr=0.01))

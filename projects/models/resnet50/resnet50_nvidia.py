_base_ = [
    'mmcls::_base_/schedules/imagenet_bs2048_coslr.py',
    'mmcls::_base_/models/resnet50.py',
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::_base_/default_runtime.py',
]

# batch size
batch_size = 256
train_dataloader = dict(batch_size=batch_size, )
val_dataloader = dict(batch_size=batch_size, )

# disable bn weight decay
paramwise_cfg = dict(custom_keys={
    'mutable_channel': dict(decay_mult=0.0),
    'bn': dict(decay_mult=0.0)
})
# turn lr
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.256 * 8,
        momentum=0.875,
        weight_decay=3.0517578125e-05,
        nesterov=False),
    paramwise_cfg=paramwise_cfg)

# change epoch
epoch = 256
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1 / 8,
        by_epoch=True,
        begin=0,
        # about 2500 iterations for ImageNet-1k
        end=8,
        # update by iter
        convert_to_iter_based=False),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=epoch - 8,
        by_epoch=True,
        begin=8,
        end=epoch,
    )
]

train_cfg = dict(max_epochs=epoch)

# add label smooth
model = dict(
    head=dict(
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0), ))

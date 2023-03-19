_base_ = [
    '../cifar10/cifar10_bs16.py', '../cifar10/cifar10_bs128_300.py',
    '../cifar10/default_runtime.py'
]
model = dict(
    _delete_=True,
    type='mmrazor.ResNetCifarSuper',
    ratio=1.5,
    num_blocks=[12, 12, 12],
)

_base_ = '../../../../../models/vgg/configs/vgg_pretrain.py'
custom_imports = dict(imports=['projects'])

pretrained_path = './work_dirs/pretrained/vgg_pretrained.pth'  # noqa

architecture = _base_.model
architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture.update({'data_preprocessor': _base_.data_preprocessor})
data_preprocessor = None

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='ChexAlgorithm',
    architecture=architecture,
    mutator_cfg=dict(
        type='ChexMutator',
        channel_unit_cfg=dict(
            type='ChexUnit', default_args=dict(choice_mode='number', )),
        channel_ratio=0.3,
    ),
    delta_t=1,
    total_steps=100,
    init_growth_rate=0.3,
)
custom_hooks = [
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=100,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=[1, 3, 32, 32],
        ),
        save_ckpt_delta_thr=[],
        early_stop=False,
    ),
]

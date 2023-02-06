#############################################################################
# where you have to fill
_base_ = ['../../../../models/vgg/configs/vgg_pretrain.py']
# pretrained_path = './work_dirs/pretrained/vgg_pretrained.pth'  # path of pretrained model # noqa

input_shape = [1, 3, 32, 32]
##############################################################################

custom_imports = dict(imports=['projects'])

architecture = _base_.model

if hasattr(_base_, 'data_preprocessor'):
    architecture.update({'data_preprocessor': _base_.data_preprocessor})
    data_preprocessor = None

# architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture['_scope_'] = _base_.default_scope

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='ChexRAlgorithm',
    architecture=architecture,
    mutator_cfg=dict(
        type='ChexRMutator',
        channel_unit_cfg=dict(
            type='ChexUnit', default_args=dict(choice_mode='number', )),
        channel_ratio=0.77,
        reallocate=False,
    ),
    delta_t=1,
    total_steps=60,
    init_growth_rate=0.3,
    model_expand_ratio=0.3,
)

custom_hooks = getattr(_base_, 'custom_hooks', []) + [
    dict(type='mmrazor.PruningStructureHook'),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=1,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=input_shape,
        ),
        early_stop=False,
        save_ckpt_thr=[],
    ),
]

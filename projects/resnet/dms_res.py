from mmrazor.implementations.pruning.dms.core.algorithm import BaseDTPAlgorithm

from mmrazor.implementations.pruning.dms.core.models.resnet_img import ResLayer, BaseBottleneck, Bottleneck

from mmrazor.registry import MODELS


@MODELS.register_module()
class ResDmsAlgo(BaseDTPAlgorithm):
    default_mutator_kwargs = dict(
        prune_qkv=False,
        prune_block=True,
        dtp_mutator_cfg=dict(
            type='DTPAMutator',
            channel_unit_cfg=dict(
                type='DTPTUnit', default_args=dict(extra_mapping={})),
            parse_cfg=dict(
                _scope_='mmrazor',
                type='ChannelAnalyzer',
                demo_input=dict(
                    type='DefaultDemoInput',
                    input_shape=(1, 3, 224, 224),
                ),
                tracer_type='FxTracer'),
        ),
        block_initilizer_kwargs=dict(
            stage_mixin_layers=[ResLayer],
            dynamic_block_mapping={BaseBottleneck: Bottleneck}),
    )

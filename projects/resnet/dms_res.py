from mmrazor.implementations.pruning.dms.core.algorithm import BaseDTPAlgorithm, DmsSubModel

from mmrazor.implementations.pruning.dms.core.models.resnet_img import ResLayer, BaseBottleneck, Bottleneck, ResNetDMS
from mmrazor.implementations.pruning.dms.core.utils import MyDropPath
from mmrazor.registry import MODELS
import torch
import torch.nn as nn
from mmrazor.utils import print_log


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


@MODELS.register_module()
def ResNetDmsSubModel(
    algorithm: BaseDTPAlgorithm,
    reset_params=False,
    drop_path_rate=-1,
):
    model = DmsSubModel(algorithm, reset_params=reset_params)
    if drop_path_rate != -1:
        backbone: ResNetDMS = model.backbone
        total_depth = sum([
            len(stage) for stage in [
                backbone.layer1, backbone.layer2, backbone.layer3,
                backbone.layer4
            ]
        ])

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        i = 0
        for stage in [
                backbone.layer1, backbone.layer2, backbone.layer3,
                backbone.layer4
        ]:
            for block in stage:
                block.drop_path = MyDropPath(
                    dpr[i]) if dpr[i] > 0 else nn.Identity()
                i += 1
        print_log(f'static model after reset drop path: {model}', )
    return model
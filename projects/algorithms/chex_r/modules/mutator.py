from mmrazor.registry import MODELS
from ...chex.modules.chex_mutator import ChexMutator  # type: ignore


@MODELS.register_module()
class ChexRMutator(ChexMutator):

    def __init__(self,
                 channel_unit_cfg={},
                 parse_cfg=dict(
                     type='ChannelAnalyzer',
                     demo_input=(1, 3, 224, 224),
                     tracer_type='BackwardTracer'),
                 channel_ratio=0.7,
                 reallocate=True,
                 init_cfg=None) -> None:
        super().__init__(channel_unit_cfg, parse_cfg, None, channel_ratio,
                         init_cfg)
        self.reallocate = reallocate

    def _get_choices_by_bn_imp(self, remain_ratio=0.5):
        if self.reallocate is True:
            return super()._get_choices_by_bn_imp(remain_ratio)
        else:
            choices = {}
            for unit in self.mutable_units:
                choices[unit.name] = int(unit.num_channels * remain_ratio)
            return choices

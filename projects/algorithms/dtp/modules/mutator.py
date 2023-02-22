import torch
from torch import nn

from mmrazor.models.mutators import ChannelMutator
from mmrazor.models.task_modules.demo_inputs import DefaultDemoInput
from mmrazor.registry import MODELS, TASK_UTILS
from .ops import QuickFlopMixin
from .unit import ImpUnit


@MODELS.register_module()
class ImpMutator(ChannelMutator[ImpUnit]):

    def __init__(
        self,
        channel_unit_cfg=dict(
            type='ImpUnit', default_args=dict(
                imp_type='l1',
                grad_clip=-1,
            )),
        parse_cfg=dict(
            _scope_='mmrazor',
            type='ChannelAnalyzer',
            demo_input=dict(
                type='DefaultDemoInput',
                input_shape=[1, 3, 224, 224],
            ),
            tracer_type='BackwardTracer',
        )
    ) -> None:
        super().__init__(channel_unit_cfg, parse_cfg)
        self.demo_input = parse_cfg['demo_input']

    def prepare_from_supernet(self, supernet) -> None:
        res = super().prepare_from_supernet(supernet)
        for unit in self.mutable_units:
            unit.activate_grad()
        self.init_quick_flop(supernet)
        return res

    @torch.no_grad()
    def resort(self):
        for unit in self.mutable_units:
            unit.resort()

    def limit_value(self):
        for unit in self.mutable_units:
            unit.mutable_channel.limit_value()

    def save_info(self):
        for unit in self.mutable_units:
            unit.mutable_channel.save_info()

    # soft flops

    @torch.no_grad()
    def init_quick_flop(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, QuickFlopMixin):
                module.start_record()
        demo_input: DefaultDemoInput = TASK_UTILS.build(self.demo_input)
        model.eval()
        input = demo_input.get_data(model, training=False)
        input['mode'] = 'tensor'
        model(**input)
        for module in model.modules():
            if isinstance(module, QuickFlopMixin):
                module.end_record()

    def get_soft_flop(self, model):
        flop = 0
        for _, module in model.named_modules():
            if isinstance(module, QuickFlopMixin):
                flop += module.soft_flop()
        assert isinstance(flop, torch.Tensor)
        return flop

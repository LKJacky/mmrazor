# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel

from mmrazor.models.task_modules.demo_inputs import BaseDemoInput
from mmrazor.registry import TASK_UTILS


@TASK_UTILS.register_module()
class OptDemoInput(BaseDemoInput):

    def _get_data(self, model, input_shape, training):
        data = dict(
            input_ids=torch.randint(0, 50265, input_shape).long(),
            attention_mask=None,
            head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False)
        return data


class SelfDistillAlgorithm(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        out = self.model(input_ids, attention_mask, position_ids,
                         past_key_values, inputs_embeds, labels, use_cache,
                         output_attentions, output_hidden_states, return_dict)
        return out

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


if __name__ == '__main__':
    model: OPTModel = OPTForCausalLM.from_pretrained('facebook/opt-125m').model

    from mmrazor.implementations.pruning.dms.core.mutator import DMSMutator
    mutator = DMSMutator(
        dtp_mutator_cfg=dict(
            type='DTPAMutator',
            channel_unit_cfg=dict(type='DTPTUnit', default_args=dict()),
            parse_cfg=dict(
                _scope_='mmrazor',
                type='ChannelAnalyzer',
                demo_input=dict(
                    type='OptDemoInput',
                    input_shape=(1, 128),
                ),
                tracer_type='FxTracer'),
        ), )
    mutator.prepare_from_supernet(model)
    mutator.init_quick_flop(model)
    assert len(mutator.dtp_mutator.mutable_units) == 13
    print(mutator.info())
    print(model)
    x = torch.rand([1, 128]).long()
    y = model(x)
    print(y['last_hidden_state'].shape)

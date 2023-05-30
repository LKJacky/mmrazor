# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel


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
    mutator = DMSMutator()
    mutator.prepare_from_supernet(model)
    # print(len(mutator.info()))
    print(model)
    x = torch.rand([1, 128]).long()
    y = model(x)
    print(y['last_hidden_state'].shape)

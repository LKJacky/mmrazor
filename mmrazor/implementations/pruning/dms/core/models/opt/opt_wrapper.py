# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json

import torch
import torch.nn as nn
from transformers import TrainingArguments
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel
from transformers.trainer_callback import (TrainerCallback, TrainerControl,
                                           TrainerState)

from mmrazor.implementations.pruning.dms.core.mutator import DMSMutator
from mmrazor.implementations.pruning.dms.core.scheduler import DMSScheduler
from mmrazor.models.task_modules.demo_inputs import BaseDemoInput
from mmrazor.registry import MODELS, TASK_UTILS
from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                  load_fix_subnet)
from mmrazor.utils import print_log


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


def to_static_model(
    algorithm,
    reset_params=False,
    **kargs,
):

    pruning_structure = algorithm.mutator.choice_template
    print_log('PruneSubModel get pruning structure:')
    print_log(json.dumps(pruning_structure, indent=4))

    # to static model
    fix_mutable = export_fix_subnet(algorithm.architecture)[0]
    load_fix_subnet(algorithm.architecture, fix_mutable)
    model = algorithm.architecture

    print_log(model)
    if reset_params:
        print_log('reset parameters')
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    return model


def convert_float_to_tenosr(res: dict, device):
    for k in res:
        if not isinstance(res[k], torch.Tensor):
            res[k] = torch.tensor(res[k], device=device)
    return res


class DmsOptAlgorithm(nn.Module):

    def __init__(self, model: OPTForCausalLM) -> None:
        super().__init__()
        self.model = copy.deepcopy(model)

        self.mutator: DMSMutator = MODELS.build(
            dict(
                type='DMSMutator',
                prune_qkv=False,
                dtp_mutator_cfg=dict(
                    type='DTPAMutator',
                    channel_unit_cfg=dict(
                        type='DTPTUnit', default_args=dict()),
                    parse_cfg=dict(
                        _scope_='mmrazor',
                        type='ChannelAnalyzer',
                        demo_input=dict(
                            type='OptDemoInput',
                            input_shape=(1, 128),
                        ),
                        tracer_type='FxTracer'),
                ),
            ))
        self.scheduler = DMSScheduler(
            self.model.model,
            self.mutator,
            flops_target=0.5,
            decay_ratio=0.8,
            refine_ratio=0.2,
            flop_loss_weight=100,
            structure_log_interval=100,
            by_epoch=False,
            target_scheduler='cos',
        )
        self.model.load_state_dict(
            model.state_dict(), strict=False)  # remain a bug

        self.runtime_info = None

        print(self.model)

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
        if self.training:
            self.scheduler.before_train_forward(*self.runtime_info)
        out = self.model(input_ids, attention_mask, position_ids,
                         past_key_values, inputs_embeds, labels, use_cache,
                         output_attentions, output_hidden_states, return_dict)
        if self.training:
            if self.runtime_info is not None:
                extra_dict = self.scheduler.after_train_forward(
                    *self.runtime_info)
                extra_dict = convert_float_to_tenosr(extra_dict,
                                                     input_ids.device)
                out['loss'] = out['loss'] + extra_dict['flops_loss']
        return out

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class DmsCallbacks(TrainerCallback):

    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                      control: TrainerControl, **kwargs):

        model: DmsOptAlgorithm = kwargs['model']
        model.runtime_info = [
            state.global_step, state.epoch, state.max_steps,
            state.num_train_epochs
        ]
        if state.global_step % model.scheduler.structure_log_interval == 0:
            print_log(model.mutator.info())
            print(model.scheduler.current_target(*model.runtime_info))


if __name__ == '__main__':

    model: OPTModel = OPTForCausalLM.from_pretrained('facebook/opt-125m').model

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
    assert len(mutator.dtp_mutator.mutable_units) == 12
    print(mutator.info())
    print(model)
    x = torch.rand([1, 128]).long()
    y = model(x)
    print(y['last_hidden_state'].shape)

    model: OPTForCausalLM = OPTForCausalLM.from_pretrained('facebook/opt-125m')
    algorithm = DmsOptAlgorithm(model)
    y = model(x)

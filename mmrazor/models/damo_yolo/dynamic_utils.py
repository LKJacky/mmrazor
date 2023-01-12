# Copyright (c) OpenMMLab. All rights reserved.

import copy

import torch.nn as nn
from mmengine import fileio

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.mutables import BaseMutable, DerivedMutable
from mmrazor.models.mutables import MutableValue as BaseMutableValue
from mmrazor.registry import MODELS

OP_MAPPING = {
    nn.Conv2d: dynamic_ops.DynamicConv2d,
    nn.BatchNorm2d: dynamic_ops.DynamicBatchNorm2d,
    nn.Linear: dynamic_ops.DynamicLinear,
    nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
    nn.Sequential: dynamic_ops.DynamicSequential
}


class MutableValue(BaseMutableValue):
    # disable choice check of BaseMutableValue
    @property
    def current_choice(self):
        return super().current_choice

    @current_choice.setter
    def current_choice(self, value):
        self._current_choice = value

    def fix_chosen(self, chosen) -> None:
        self.current_choice = chosen
        self.is_fixed = True


class MutableDepth(MutableValue):
    pass


class DynamicOneshotModule(nn.ModuleDict, dynamic_ops.DynamicMixin):
    accepted_mutable_attrs = {'module_type'}

    def __init__(self, modules=None) -> None:
        super().__init__(modules)
        self.mutable_attrs: nn.ModuleDict = nn.ModuleDict()

    def register_mutable_attr(self, attr: str, mutable: BaseMutable):
        assert attr == 'module_type'
        self.mutable_attrs[attr] = mutable

    def forward(self, x):
        module = self.forward_module
        if module is not None:
            return module(x)
        else:
            return x

    @property
    def forward_module(self):
        if len(list(self.pure_children())) == 0:
            return None
        else:
            return self[self.module_type]

    @property
    def module_type(self):
        if 'module_type' in self.mutable_attrs:
            return self.mutable_attrs['module_type'].current_choice
        else:
            key = list(self.named_pure_children())[0][0]
            return key

    @classmethod
    def convert_from(cls, module):
        return super().convert_from(module)

    def static_op_factory(self):
        return super().static_op_factory

    def named_pure_children(self):
        for name, child in self.named_children():
            if name != 'mutable_attrs':
                yield name, child

    def pure_children(self):
        for _, child in self.named_pure_children():
            yield child

    def to_static_op(self) -> nn.Module:
        # statc_op = nn.Sequential()
        module = self.forward_module
        return module


def replace_with_dynamic_ops(model: nn.Module, dynamicop_map=OP_MAPPING):
    """Replace torch modules with dynamic-ops."""

    def replace_op(model: nn.Module, name: str, module: nn.Module):
        names = name.split('.')
        for sub_name in names[:-1]:
            model = getattr(model, sub_name)

        setattr(model, names[-1], module)

    for name, module in model.named_modules():
        if isinstance(module, nn.Module):

            if type(module) in dynamicop_map:
                new_module = dynamicop_map[type(module)].convert_from(module)
                replace_op(model, name, new_module)


class SearchableModuleMinin():

    def init_search_space(self):
        pass

    def search_space(self):
        saved_mutable_set = set()
        return self._search_space(saved_mutables=saved_mutable_set)

    def _search_space(self: nn.Module, saved_mutables: set):
        if isinstance(self, SearchableModuleMinin):
            mutables_dict = self._mutable_children(saved_mutables)
        else:
            mutables_dict = nn.ModuleDict()

        for name, module in self.named_children():
            if isinstance(module, SearchableModuleMinin):
                sub_dict = module._search_space(saved_mutables)
                if SearchableModuleMinin.count_mutables_in_dict(sub_dict) > 0:
                    mutables_dict[name] = sub_dict
            elif isinstance(module, nn.Module):
                sub_dict = SearchableModuleMinin._search_space(
                    module, saved_mutables)
                if SearchableModuleMinin.count_mutables_in_dict(sub_dict) > 0:
                    mutables_dict[name] = sub_dict
        return mutables_dict

    def _mutable_children(self: nn.Module, saved_mutables: set):
        mutables_dict = nn.ModuleDict()
        for name, module in self.named_children():
            if isinstance(module, BaseMutable) and not isinstance(
                    module, DerivedMutable):
                if module not in saved_mutables:
                    saved_mutables.add(module)
                    mutables_dict[name] = module
        return mutables_dict

    @staticmethod
    def count_mutables_in_dict(mutable_dict: dict):
        num = 0
        for value in mutable_dict.values():
            if isinstance(value, dict) or isinstance(value, nn.ModuleDict):
                num += SearchableModuleMinin.count_mutables_in_dict(value)
            elif isinstance(value, BaseMutable):
                num += 1
            else:
                raise NotImplementedError(f'{type(value)}')
        return num

    # dump and load

    def dump(self):
        return self._dump(self.search_space())

    @classmethod
    def _dump(cls, space):
        if isinstance(space, nn.ModuleDict):
            res = {}
            for key in space:
                res[key] = cls._dump(space[key])
        elif isinstance(space, BaseMutable):
            res = space.current_choice
        else:
            raise NotImplementedError()
        return res

    def load(self, space):
        self._load(self.search_space(), space)

    @classmethod
    def _load(cls, search_space, space):
        if isinstance(search_space, BaseMutable):
            search_space.current_choice = space

        elif isinstance(search_space, nn.ModuleDict):
            for key in space:
                cls._load(search_space[key], space[key])
        else:
            raise NotImplementedError()

    # to_static

    def to_static_op(self: nn.Module):
        for name, module in self.named_children():
            if isinstance(module, dynamic_ops.DynamicMixin):
                setattr(self, name, module.to_static_op())
                module = getattr(self, name)

            if isinstance(module, SearchableModuleMinin):
                setattr(self, name, module.to_static_op())
            elif isinstance(module, BaseMutable):
                pass
            elif isinstance(module, nn.Module):
                setattr(self, name, SearchableModuleMinin.to_static_op(module))
            else:
                raise NotImplementedError()
        SearchableModuleMinin.delete_mutables(self)
        return self

    def delete_mutables(self: nn.Module):
        for name, module in copy.copy(list(self.named_children())):
            if isinstance(module, BaseMutable):
                delattr(self, name)


@MODELS.register_module()
def SearchAableModelDeployWrapper(architecture,
                                  subnet_dict=None,
                                  to_static=True):
    if isinstance(architecture, dict):
        architecture = MODELS.build(architecture)
    if isinstance(architecture, SearchableModuleMinin):
        architecture.init_search_space()

    if isinstance(subnet_dict, str):
        subnet_dict = fileio.load(subnet_dict)
    if subnet_dict is not None:
        architecture.load(subnet_dict)
    import json
    print(json.dumps(architecture.dump(), indent=4))
    if to_static:
        # subnet = export_fix_subnet(architecture)[0]
        # load_fix_subnet(architecture, subnet)
        architecture = architecture.to_static_op()
    return architecture

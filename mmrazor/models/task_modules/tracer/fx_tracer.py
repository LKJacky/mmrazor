# Copyright (c) OpenMMLab. All rights reserved.
"""This module define FxTracer and related classes."""

import functools
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.fx as fx
import torch.nn as nn
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch.fx._symbolic_trace import (Tracer, _autowrap_check,
                                      _orig_module_call, _orig_module_getattr,
                                      _patch_wrapped_functions, _Patcher)
from torch.fx.graph import Graph
from torch.fx.node import Argument
from torch.fx.node import Node as FxNode
from torch.fx.proxy import Proxy

from mmrazor.registry import TASK_UTILS
from mmrazor.structures.graph.base_graph import BaseGraph, BaseNode


class FxBaseNode(BaseNode):
    """Node to record FxNode."""

    def __init__(self, name: str, val: FxNode) -> None:
        super().__init__(name, val)

    def module(self):
        """Union[Module | None]: the module the fxnode corresponding to."""
        self.val: FxNode
        model = self.val.graph.owning_module
        if self.val.op == 'call_module':
            target = self.val.target
            target = target.split('.')
            obj = model
            for t in target:
                obj = getattr(obj, t)
            return obj
        else:
            return None

    def function(self):
        """Union[Callable | Node]: the function the fxnode corresponding to."""
        if self.is_function():
            return self.val.target
        else:
            return None

    # base type
    # placeholder|call_method|call_module|call_function|get_attr|output

    def is_function(self):
        """Bool: if the fxnode represents 'call_function'"""
        return self.val.op == 'call_function'

    def is_module(self):
        """Bool: if the fxnode represents 'call_module'"""
        return self.val.op == 'call_module'

    def is_Tensor(self):
        """Bool: if the fxnode represents input or output tensors"""
        return self.val.op == 'placeholder' or self.val.op == 'output'

    def is_method(self):
        """Bool: if the fxnode represents 'call_method'"""
        return self.val.op == 'call_method'

    def is_get_attr(self):
        """Bool: if the fxnode represents 'get_attr'"""
        return self.val.op == 'get_attr'

    # extended type

    def is_cat(self):
        """Bool: if the fxnode represents a cat node"""
        return self.is_function() and self.function() is torch.cat

    # other

    def __repr__(self) -> str:
        return f'{self.name}({self.val.op})'


class CostumTracer(Tracer):
    """CostumTracer allow user to indicate leaf module."""

    def __init__(self,
                 is_extra_leaf_module: Callable[[nn.Module, str], bool] = None,
                 warp_method={},
                 concrete_args={}) -> None:
        """
        Args:
            is_extra_leaf_module: Callable[[nn.Module, str], bool]: a function
            to determine if a module is a leaf module except torch pre-defined
            modules.
        """
        super().__init__(
            param_shapes_constant=True,
            autowrap_functions=[torch.arange],
        )
        self.extra_is_leaf_module = is_extra_leaf_module
        self.concrete_args = concrete_args
        from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
        from mmdet.models.dense_heads.rpn_head import RPNHead
        self.warp_method = {
            torch: torch.arange,
            RPNHead: RPNHead.predict_by_feat,
            BaseDenseHead: BaseDenseHead.predict_by_feat,
        }

    def is_leaf_module(self, m: torch.nn.Module,
                       module_qualified_name: str) -> bool:
        """Bool: determine if a module is a leaf module"""
        is_torch_module = super().is_leaf_module(m, module_qualified_name)
        if self.extra_is_leaf_module is None:
            is_extra = False
        else:
            is_extra = self.extra_is_leaf_module(m, module_qualified_name)
        return is_torch_module or is_extra

    def trace(self, root) -> fx.graph.Graph:
        return self._trace(root, self.concrete_args)

    def _trace(self,
               root: Union[torch.nn.Module, Callable[..., Any]],
               concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
        if isinstance(root, torch.nn.Module):
            self.root = root

            assert hasattr(type(root), self.traced_func_name), (
                f"traced_func_name={self.traced_func_name} doesn't exist in"
                ' {type(root).__name__}')

            fn = getattr(type(root), self.traced_func_name)
            self.submodule_paths = {
                mod: name
                for name, mod in root.named_modules()
            }
        else:
            self.root = torch.nn.Module()
            fn = root

        tracer_cls: Optional[Type['Tracer']] = getattr(self, '__class__', None)
        self.graph = Graph(tracer_cls=tracer_cls)

        # When we encounter a Tensor value that's not a parameter, we look
        # if it
        # is some other attribute on the model. Construct a dict mapping
        # Tensor
        # values to the qualified name here for efficiency. This is used
        # downstream
        # in create_arg
        self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

        def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
            for k, v in m.__dict__.items():
                if isinstance(v, (torch.Tensor, ScriptObject)):
                    self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
            for k, v in m.named_children():
                collect_tensor_attrs(v, prefix_atoms + [k])

        collect_tensor_attrs(self.root, [])

        assert isinstance(fn, FunctionType)

        fn_globals = fn.__globals__  # run before it gets patched
        fn, args = self.create_args_for_root(fn,
                                             isinstance(root, torch.nn.Module),
                                             concrete_args)

        parameter_proxy_cache: Dict[str, Proxy] = {
        }  # Reduce number of get_attr calls

        # Method dispatch on parameters is not recorded unless it's directly
        # used.
        # Thus, we need to insert a proxy when __getattr__ requests a
        # parameter.
        @functools.wraps(_orig_module_getattr)
        def module_getattr_wrapper(mod, attr):
            attr_val = _orig_module_getattr(mod, attr)
            return self._module_getattr(attr, attr_val, parameter_proxy_cache)

        @functools.wraps(_orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):

            def forward(*args, **kwargs):
                return _orig_module_call(mod, *args, **kwargs)

            _autowrap_check(
                patcher,
                getattr(getattr(mod, 'forward', mod), '__globals__', {}),
                self._autowrap_function_ids)
            return self.call_module(mod, forward, args, kwargs)

        with _Patcher() as patcher:
            # allow duplicate patches to support the case of nested calls
            patcher.patch_method(
                torch.nn.Module,
                '__getattr__',
                module_getattr_wrapper,
                deduplicate=False)
            patcher.patch_method(
                torch.nn.Module,
                '__call__',
                module_call_wrapper,
                deduplicate=False)
            for obj, mth in self.warp_method.items():
                patcher.patch_method(
                    obj,
                    mth.__name__,
                    self.warp_a_method(obj, mth),
                    deduplicate=False)
            _patch_wrapped_functions(patcher)
            _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
            for module in self._autowrap_search:
                _autowrap_check(patcher, module.__dict__,
                                self._autowrap_function_ids)
            self.create_node(
                'output',
                'output', (self.create_arg(fn(*args)), ), {},
                type_expr=fn.__annotations__.get('return', None))

        self.submodule_paths = None

        return self.graph

    def call_method(self, obj, origin_fn, name, args, kwargs):
        return self.create_proxy('call_function', origin_fn, args, kwargs,
                                 name)

    def warp_a_method(self, obj, origin_fn):

        @functools.wraps(origin_fn)
        def fn_wrapper(*args, **kwargs):
            return self.call_method(obj, origin_fn, origin_fn.__name__, args,
                                    kwargs)

        return fn_wrapper

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        try:
            proxy = super().call_module(m, forward, args, kwargs)
            return proxy
        except Exception:
            module_qualified_name = self.path_of_module(m)
            from mmengine import MMLogger
            MMLogger.get_current_instance().warning(
                f'{module_qualified_name}({type(m)}) encounter error when'
                ' tracing. '
                'It will be treated as a leaf module.')
            return self.create_proxy('call_module', module_qualified_name,
                                     args, kwargs)

    def create_arg(self, a: Any) -> 'Argument':
        try:
            arg = super().create_arg(a)
            return arg
        except Exception:
            return a


@TASK_UTILS.register_module()
class RazorFxTracer(CostumTracer):
    """A wapper for torch.fx.tracer."""

    def __init__(self,
                 is_extra_leaf_module: Callable[[nn.Module, str], bool] = None,
                 concrete_args={}) -> None:
        if isinstance(is_extra_leaf_module, dict):
            is_extra_leaf_module = TASK_UTILS.build(is_extra_leaf_module)
        super().__init__(
            is_extra_leaf_module=is_extra_leaf_module,
            concrete_args=concrete_args)

    def add_node(self, graph: BaseGraph[FxBaseNode], fxnode: FxNode):
        """FxBaseNode: convert a torch FxNode to a FxBaseNode, and add it the
        self.graph"""
        node = graph.add_or_find_node(FxBaseNode(fxnode.name, fxnode))
        return node

    def parse_torch_graph(self, torch_graph: fx.graph.Graph):
        """None: convert torch graph to self.graph"""

        graph = BaseGraph[FxBaseNode]()
        # copy_nodes
        for fxnode in torch_graph.nodes:
            self.add_node(graph, fxnode)

        # connect nodes
        for fxnode in torch_graph.nodes:
            for pre_node in fxnode.all_input_nodes:
                graph.connect(
                    self.add_node(graph, pre_node),
                    self.add_node(graph, fxnode))

        return graph

    def trace(self, model) -> BaseGraph[FxBaseNode]:
        torch_graph = super().trace(model)
        torch_graph.owning_module = model

        self.graph = BaseGraph[FxBaseNode]()
        return self.parse_torch_graph(torch_graph)

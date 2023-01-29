# Copyright (c) OpenMMLab. All rights reserved.

from mmrazor.registry import TASK_UTILS
from ....cores.counters import DynamicConv2dCounter  # type: ignore
from ....cores.counters import DynamicLinearCounter  # type: ignore


@TASK_UTILS.register_module()
class ChexLinearCounter(DynamicLinearCounter):
    pass


@TASK_UTILS.register_module()
class ChexConv2Counter(DynamicConv2dCounter):
    pass

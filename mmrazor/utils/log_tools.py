# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmengine import MMLogger, dist
from mmengine import print_log as engine_print_log


def get_level(level='info'):
    if isinstance(level, str):
        level = level.upper()
        assert level in logging._nameToLevel
        level = logging._nameToLevel[level]
    elif isinstance(level, int):
        pass
    else:
        raise NotImplementedError()
    return level


def print_log(msg, logger='current', level='info', only_rank_zero=True):
    should_print = True
    if only_rank_zero:
        if dist.get_rank() != 0:
            should_print = False
    if should_print:
        engine_print_log(msg, logger, get_level(level))


def set_log_level(level='debug'):
    level = get_level(level)

    logger = MMLogger.get_current_instance()
    logger.handlers[0].setLevel(level)
    logger.setLevel(level)

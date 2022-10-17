# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List
from typing import Dict, Callable
from mmrazor.registry import MODELS
from mmengine.config import Config
import os
from mmengine.utils import get_installed_path
from mmrazor.registry import MODELS
import torch
import torch.nn as nn


class ModelLibrary:
    default_includes: List = []

    def __init__(self, include=default_includes, exclude=[]) -> None:
        self.include_key = include
        self.exclude_key = exclude
        self.models: Dict[str, Callable] = self.get_models()

    def get_models(self):
        raise NotImplementedError()

    def include_models(self):
        models = []
        for name in self.models:
            if self.is_include(name, self.include_key)\
                    and not self.is_exclude(name, self.exclude_key):
                models.append(self.models[name])
        return models

    def is_include(self, name: str, includes: List[str]):
        for key in includes:
            if name.startswith(key):
                return True
        return False

    def is_exclude(self, name, excludes):
        for key in excludes:
            if key in name:
                return True
        return False

    def is_default_includes_cover_all_models(self):
        models = copy.copy(self.models)
        is_covered = True
        for name in models:
            if self.is_include(name, self.__class__.default_includes):
                pass
            else:
                is_covered = False
                print(name, '\tnot include')
        return is_covered


class TorchModelLibrary(ModelLibrary):

    default_includes = [
        'alexnet', 'densenet', 'efficientnet', 'googlenet', 'inception',
        'mnasnet', 'mobilenet', 'regnet', 'resnet', 'resnext', 'shufflenet',
        'squeezenet', 'vgg', 'wide_resnet', "vit", "swin", "convnext"
    ]

    def __init__(self, include=default_includes, exclude=[]) -> None:
        super().__init__(include, exclude)

    def get_models(self):
        from inspect import isfunction

        import torchvision

        attrs = dir(torchvision.models)
        models = {}
        for name in attrs:
            module = getattr(torchvision.models, name)
            if isfunction(module) and name is not 'get_weight':
                models[name] = module
        return models


def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = nn.BatchNorm2d
        module_output = nn.BatchNorm2d(module.num_features, module.eps,
                                       module.momentum, module.affine,
                                       module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


class MMModelGenerator:

    def __init__(self, name, cfg) -> None:
        self.name = name
        self.cfg = cfg

    def __call__(self):
        model = MODELS.build(self.cfg)
        model = revert_sync_batchnorm(model)
        return model

    def __repr__(self) -> str:
        return self.name


class MMModelLibrary(ModelLibrary):
    default_includes = []

    def __init__(self,
                 repo='mmcls',
                 model_config_path='_base_/models/',
                 include=default_includes,
                 exclude=[]) -> None:
        self.config_path = self._get_model_config_path(repo, model_config_path)
        self.repo = repo
        super().__init__(include, exclude)

    def get_models(self):
        models = {}
        for dirpath, dirnames, filenames in os.walk(self.config_path):
            for filename in filenames:
                if filename.endswith('.py'):
                    model_type_name = '_'.join(
                        dirpath.replace(self.config_path, '').split('/'))
                    model_type_name = model_type_name if model_type_name == '' else model_type_name + '_'
                    model_name = model_type_name + \
                        os.path.basename(filename).split('.')[0]

                    cfg_path = dirpath + '/' + filename
                    model_cfg = Config.fromfile(cfg_path)['model']
                    model_cfg = self._config_process(model_cfg)
                    models[model_name] = MMModelGenerator(
                        model_name, model_cfg)
        return models

    def get_default_model_names(self):

        def get_base_model_name(name: str):
            names = name.split('_')
            return names[0]

        names = []
        for name in self.models:
            base_name = get_base_model_name(name)
            if base_name not in names:
                names.append(base_name)

        return names

    def _get_model_config_path(self, repo, config_path):
        repo_path = get_installed_path(repo)
        return repo_path + '/.mim/configs/' + config_path

    def _config_process(self, config: Dict):
        config['_scope_'] = self.repo
        return config


class MMClsModelLibrary(MMModelLibrary):
    default_includes = [
        'vgg',
        'efficientnet',
        'resnet',
        'mobilenet',
        'resnext',
        'wide-resnet',
        'shufflenet',
        'hrnet',
        'resnest',
        'inception',
        'res2net',
        'densenet',
        'convnext',
        'regnet',
        'van',
        'swin_transformer',
        'convmixer',
        't2t',
        'twins',
        'repmlp',
        'tnt',
        't2t',
        'mlp_mixer',
        'conformer',
        'poolformer',
        'vit',
        'efficientformer',
        'mobileone',
        'edgenext',
        'mvit',
        'seresnet',
        'repvgg',
        'seresnext',
        'seresnext',
    ]

    def __init__(self, include=default_includes, exclude=[]) -> None:
        super().__init__(
            repo='mmcls',
            model_config_path='_base_/models/',
            include=include,
            exclude=exclude)


class MMDetModelLibrary(MMModelLibrary):

    default_includes = [
        'rpn',
        'faster-rcnn',
        'cascade-rcnn',
        'fast-rcnn',
        'cascade-mask-rcnn',
        'retinanet',
        'mask-rcnn',
        'ssd300',
    ]

    def __init__(self, include=default_includes, exclude=[]) -> None:
        super().__init__(
            repo='mmdet',
            model_config_path='_base_/models/',
            include=include,
            exclude=exclude)

    def _config_process(self, config: Dict):
        config = super()._config_process(config)
        if 'preprocess_cfg' in config:
            config.pop('preprocess_cfg')
        if 'pretrained' in config:
            config.pop('pretrained')
        return config


class MMSegModelLibrary(MMModelLibrary):
    default_includes: List = [
        'cgnet',
        'gcnet',
        'setr',
        'deeplabv3',
        'twins',
        'fastfcn',
        'fpn',
        'upernet',
        'dnl',
        'icnet',
        'segmenter',
        'encnet',
        'erfnet',
        'segformer',
        'apcnet',
        'fast',
        'ocrnet',
        'lraspp',
        'dpt',
        'fcn',
        'psanet',
        'bisenetv2',
        'pointrend',
        'ccnet',
        'pspnet',
        'dmnet',
        'stdc',
        'ann',
        'nonlocal',
        'isanet',
        'danet',
        'emanet',
        'deeplabv3plus',
        'bisenetv1',
    ]

    def __init__(self, include=default_includes, exclude=[]) -> None:
        super().__init__('mmsegmentation', '_base_/models/', include, exclude)

    def _config_process(self, config: Dict):
        config['_scope_'] = 'mmseg'
        return config
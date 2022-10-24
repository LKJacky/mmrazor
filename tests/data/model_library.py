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
from .models import (AddCatModel, ConcatModel, ConvAttnModel, DwConvModel,
                     ExpandLineModel, GroupWiseConvModel, SingleLineModel,
                     MultiBindModel, MultiConcatModel, MultiConcatModel2,
                     ResBlock, Xmodel, MultipleUseModel, Icep, SelfAttention)
import json
# model generator


class ModelGenerator:

    def __init__(self, name: str, model_src) -> None:
        self.name = name
        self.model_src = model_src

    def __call__(self, ) -> nn.Module:
        return self.model_src()

    def __repr__(self) -> str:
        return self.name


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


# model library


class ModelLibrary:
    default_includes: List = []

    def __init__(self, include=default_includes, exclude=[]) -> None:
        self.include_key = include
        self.exclude_key = exclude
        self.models: Dict[str, Callable] = self.get_models()
        self._include_models, self._uninclude_models, self.exclude_models =\
             self._classify_models(self.models)

    def get_models(self):
        raise NotImplementedError()

    def include_models(self):
        return self._include_models

    def uninclude_models(self):
        return self._uninclude_models

    def is_include(self, name: str, includes: List[str], start_with=False):
        for key in includes:
            if start_with:
                if name.startswith(key):
                    return True
            else:
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

    def short_names(self):

        def get_short_name(name: str):
            names = name.replace('-', '.').replace('_', '.').split('.')
            return names[0]

        short_names = set()
        for name in self.models:
            short_names.add(get_short_name(name))
        return short_names

    def _classify_models(self, models: Dict):
        include = []
        uninclude = []
        exclude = []
        for name in models:
            if self.is_include(name, self.exclude_key):
                exclude.append(models[name])
            elif self.is_include(name, self.include_key):
                include.append(models[name])
            else:
                uninclude.append(models[name])
        return include, uninclude, exclude


class DefaultModelLibrary(ModelLibrary):

    default_includes: List = [
        'SingleLineModel',
        'ResBlock',
        'AddCatModel',
        'ConcatModel',
        'MultiConcatModel',
        'MultiConcatModel2',
        'GroupWiseConvModel',
        'Xmodel',
        'MultipleUseModel',
        'Icep',
        'ExpandLineModel',
        'MultiBindModel',
        'DwConvModel',
        'ConvAttnModel',
        'SelfAttention',
    ]
    def __init__(self, include=default_includes, exclude=[]) -> None:
        super().__init__(include, exclude)

    def get_models(self):
        models = [
            SingleLineModel,
            ResBlock,
            AddCatModel,
            ConcatModel,
            MultiConcatModel,
            MultiConcatModel2,
            GroupWiseConvModel,
            Xmodel,
            MultipleUseModel,
            Icep,
            ExpandLineModel,
            MultiBindModel,
            DwConvModel,  #
            ConvAttnModel,
            SelfAttention,
        ]
        model_dict = {}
        for model in models:
            model_dict[model.__name__] = ModelGenerator(
                'default.' + model.__name__, model)
        return model_dict


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
                models[name] = ModelGenerator('torch.' + name, module)
        return models


class MMModelLibrary(ModelLibrary):
    default_includes = []

    def __init__(self,
                 repo='mmcls',
                 model_config_path='/',
                 include=default_includes,
                 exclude=[]) -> None:
        self.config_path = self._get_model_config_path(repo, model_config_path)
        self.repo = repo
        super().__init__(include, exclude)

    def get_models(self):
        models = {}
        added_models = set()
        for dirpath, dirnames, filenames in os.walk(self.config_path):
            for filename in filenames:
                if filename.endswith('.py'):

                    cfg_path = dirpath + '/' + filename
                    config = Config.fromfile(cfg_path)
                    if 'model' in config:

                        # get model_name
                        model_type_name = '_'.join(
                            dirpath.replace(self.config_path, '').split('/'))
                        model_type_name = model_type_name if model_type_name == '' else model_type_name + '_'
                        model_name = model_type_name + \
                            os.path.basename(filename).split('.')[0]

                        model_cfg = config['model']
                        model_cfg = self._config_process(model_cfg)
                        if json.dumps(model_cfg) not in added_models:
                            models[model_name] = MMModelGenerator(
                                self.repo + '.' + model_name, model_cfg)
                            added_models.add(json.dumps(model_cfg))
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
        config = self._remove_certain_key(config, 'init_cfg')
        config = self._remove_certain_key(config, 'pretrained')
        config = self._remove_certain_key(config, 'Pretrained')
        return config

    def _remove_certain_key(self, config: Dict, key: str = 'init_cfg'):
        if isinstance(config, dict):
            if key in config:
                config.pop(key)
            for keyx in config:
                config[keyx] = self._remove_certain_key(config[keyx], key)
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

    def __init__(self,
                 include=default_includes,
                 exclude=['cutmix', 'cifar', 'gem']) -> None:
        super().__init__(
            repo='mmcls',
            model_config_path='_base_/models/',
            include=include,
            exclude=exclude)


class MMDetModelLibrary(MMModelLibrary):

    default_includes = [
        '_base',
        'gfl',
        'sparse',
        'simple',
        'pisa',
        'lvis',
        'carafe',
        'selfsup',
        'solo',
        'ssd',
        'res2net',
        'yolof',
        'reppoints',
        'htc',
        'groie',
        'dyhead',
        'grid',
        'soft',
        'swin',
        'regnet',
        'gcnet',
        'ddod',
        'instaboost',
        'point',
        'vfnet',
        'pafpn',
        'ghm',
        'mask',
        'resnest',
        'tood',
        'detectors',
        'cornernet',
        'convnext',
        'cascade',
        'paa',
        'detr',
        'rpn',
        'ld',
        'lad',
        'ms',
        'faster',
        'centripetalnet',
        'gn',
        'dcnv2',
        'legacy',
        'panoptic',
        'strong',
        'fpg',
        'deformable',
        'free',
        'scratch',
        'openimages',
        'fsaf',
        'rtmdet',
        'solov2',
        'yolact',
        'empirical',
        'centernet',
        'hrnet',
        'guided',
        'deepfashion',
        'fast',
        'mask2former',
        'retinanet',
        'autoassign',
        'gn+ws',
        'dcn',
        'yolo',
        'foveabox',
        'libra',
        'double',
        'queryinst',
        'resnet',
        'nas',
        'sabl',
        'fcos',
        'scnet',
        'maskformer',
        'pascal',
        'cityscapes',
        'timm',
        'seesaw',
        'pvt',
        'atss',
        'efficientnet',
        'wider',
        'tridentnet',
        'dynamic',
        'yolox',
    ]

    def __init__(self,
                 include=default_includes,
                 exclude=['lad', 'ld']) -> None:
        super().__init__(
            repo='mmdet',
            model_config_path='/',
            include=include,
            exclude=exclude)

    def _config_process(self, config: Dict):
        config = super()._config_process(config)
        if 'preprocess_cfg' in config:
            config.pop('preprocess_cfg')
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


# tools


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

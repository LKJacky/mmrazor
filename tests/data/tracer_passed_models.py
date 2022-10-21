# Copyright (c) OpenMMLab. All rights reserved.
from .model_library import (MMClsModelLibrary, MMDetModelLibrary, ModelLibrary,
                            DefaultModelLibrary, TorchModelLibrary,
                            MMSegModelLibrary)
import os
from typing import List


class PassedModelManager:

    @classmethod
    def librarys(cls) -> List[ModelLibrary]:
        return []

    @classmethod
    def include_models(cls, full_test=False):
        if full_test:
            models = []
            for library in cls.librarys():
                models.extend(library.include_models())
            return models
        else:
            return cls.librarys()[0].include_models()

    @classmethod
    def uninclude_models(cls, full_test=False):
        if full_test:
            models = []
            for library in cls.librarys():
                models.extend(library.uninclude_models())
            return models
        else:
            return cls.librarys()[0].uninclude_models()


class FxPassedModelManager(PassedModelManager):

    @classmethod
    def default_library(cls):
        library = DefaultModelLibrary(include=[
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
        ])
        return library

    @classmethod
    def torch_library(cls):
        """
        googlenet: return a tuple when training, so it should
        trace in eval mode
        """
        torch_includes = [
            'alexnet',
            'densenet',
            'efficientnet',
            'googlenet',
            'inception',
            'mnasnet',
            'mobilenet',
            'regnet',
            'resnet',
            'resnext',
            # 'shufflenet', # bug
            'squeezenet',
            'vgg',
            'wide_resnet',
            # "vit",
            # "swin",
            # "convnext"
        ]
        torch_model_library = TorchModelLibrary(include=torch_includes)
        return torch_model_library

    @classmethod
    def mmcls_library(cls):
        """
        shufflenet consists of chunk operations.
        resnest: resnest has two problems. First it uses *x.shape() which is
            not tracerable using fx tracer. Second, it uses channel folding.
        res2net: res2net consists of split operations.
        convnext: consist of layernorm.
        """
        mmcls_include = [
            'vgg',
            'efficientnet',
            'resnet',
            'mobilenet',
            'resnext',
            'wide-resnet',
            # 'shufflenet', # bug
            'hrnet',
            # 'resnest',  # bug
            'inception',
            # 'res2net',  # bug
            'densenet',
            # 'convnext',  # bug
            'regnet',
            # transformer and mlp
            # # 'van', # bug
            # # 'swin_transformer', # bug
            # 'convmixer', # bug
            # # 't2t', # bug
            # # 'twins', # bug
            # # 'repmlp', # bug
            # # 'tnt', # bug
            # # 't2t', # bug
            # # 'mlp_mixer', # bug
            # # 'conformer', # bug
            # # 'poolformer', # bug
            # # 'vit', # bug
            # 'efficientformer',
            # 'mobileone',
            # 'edgenext'
        ]
        mmcls_model_library = MMClsModelLibrary(include=mmcls_include)
        return mmcls_model_library

    @classmethod
    def mmdet_library(cls):
        mmdet_include = [
            'retinanet',
            'faster-rcnn',
            'mask-rcnn',
            'fcos',
            # '_base',
            # 'gfl',
            # 'sparse',
            # 'simple',
            # 'pisa',
            # 'lvis',
            # 'carafe',
            # 'selfsup',
            # 'solo',
            # 'ssd',
            # 'res2net',
            # 'yolof',
            # 'reppoints',
            # 'htc',
            # 'groie',
            # 'dyhead',
            # 'grid',
            # 'soft',
            # 'swin',
            # 'regnet',
            # 'gcnet',
            # 'ddod',
            # 'instaboost',
            # 'point',
            # 'vfnet',
            # 'pafpn',
            # 'ghm',
            # 'mask',
            # 'resnest',
            # 'tood',
            # 'detectors',
            # 'cornernet',
            # 'convnext',
            # 'cascade',
            # 'paa',
            # 'detr',
            # 'rpn',
            # 'ld',
            # 'lad',
            # 'ms',
            # 'faster',
            # 'centripetalnet',
            # 'gn',
            # 'dcnv2',
            # 'legacy',
            # 'panoptic',
            # 'strong',
            # 'fpg',
            # 'deformable',
            # 'free',
            # 'scratch',
            # 'openimages',
            # 'fsaf',
            # 'rtmdet',
            # 'solov2',
            # 'yolact',
            # 'empirical',
            # 'centernet',
            # 'hrnet',
            # 'guided',
            # 'deepfashion',
            # 'fast',
            # 'mask2former',
            # 'retinanet',
            # 'autoassign',
            # 'gn+ws',
            # 'dcn',
            # 'foveabox',
            # 'libra',
            # 'double',
            # 'queryinst',
            # 'resnet',
            # 'nas',
            # 'sabl',
            # 'fcos',
            # 'scnet',
            # 'maskformer',
            # 'pascal',
            # 'cityscapes',
            # 'timm',
            # 'seesaw',
            # 'pvt',
            # 'atss',
            # 'efficientnet',
            # 'wider',
            # 'tridentnet',
            # 'dynamic',
            # 'yolox',
        ]
        mmdet_model_library = MMDetModelLibrary(mmdet_include)
        return mmdet_model_library

    @classmethod
    def mmseg_library(cls):
        include = [
            # 'cgnet',
            # 'gcnet',
            # 'setr',
            # 'deeplabv3',
            # 'twins',
            # 'fastfcn',
            # 'fpn',
            # 'upernet',
            # 'dnl',
            # 'icnet',
            # 'segmenter',
            # 'encnet',
            # 'erfnet',
            # 'segformer',
            # 'apcnet',
            # 'fast',
            # 'ocrnet',
            # 'lraspp',
            # 'dpt',
            # 'fcn',
            # 'psanet',
            # 'bisenetv2',
            # 'pointrend',
            # 'ccnet',
            'pspnet',
            # 'dmnet',
            # 'stdc',
            # 'ann',
            # 'nonlocal',
            # 'isanet',
            # 'danet',
            # 'emanet',
            # 'deeplabv3plus',
            # 'bisenetv1',
        ]
        model_library = MMSegModelLibrary(include=include)
        return model_library

    @classmethod
    def librarys(cls) -> List[ModelLibrary]:
        return [
            cls.default_library(),
            cls.torch_library(),
            cls.mmcls_library(),
            cls.mmseg_library(),
            cls.mmdet_library(),
        ]

    # for backward tracer


class BackwardPassedModelManager(PassedModelManager):

    @classmethod
    def default_library(cls):
        library = DefaultModelLibrary(include=[
            'LineModel',
            'ResBlock',
            'AddCatModel',
            'ConcatModel',
            'MultiConcatModel',
            'MultiConcatModel2',
            'GroupWiseConvModel',
            'Xmodel',
            # 'MultipleUseModel', # bug
            'Icep',
            'ExpandLineModel',
            'MultiBindModel',
            'DwConvModel',
            'ConvAttnModel',
        ])
        return library

    @classmethod
    def torch_library(cls):
        """
        googlenet return a tuple when training, so it
            should trace in eval mode
        """

        torch_includes = [
            'alexnet',
            'densenet',
            'efficientnet',
            'googlenet',
            'inception',
            'mnasnet',
            'mobilenet',
            'regnet',
            'resnet',
            'resnext',
            # 'shufflenet',     # bug
            'squeezenet',
            'vgg',
            'wide_resnet',
            # "vit",
            # "swin",
            # "convnext"
        ]
        torch_model_library = TorchModelLibrary(include=torch_includes)
        return torch_model_library

    @classmethod
    def mmcls_library(cls):
        """
        shufflenet consists of chunk operations.
        resnest: resnest has two problems. First it uses *x.shape() which is
            not tracerable using fx tracer. Second, it uses channel folding.
        res2net: res2net consists of split operations.
        convnext: consist of layernorm.
        """
        mmcls_model_include = [
            'vgg',
            'efficientnet',
            'resnet',
            'mobilenet',
            'resnext',
            'wide-resnet',
            # 'shufflenet',  # bug
            'hrnet',
            # 'resnest',  # bug
            'inception',
            # 'res2net',  # bug
            'densenet',
            # 'convnext',  # bug
            'regnet',
            # 'van',  # bug
            # 'swin_transformer',  # bug
            # 'convmixer', # bug
            # 't2t',  # bug
            # 'twins',  # bug
            # 'repmlp',  # bug
            # 'tnt',  # bug
            # 't2t',  # bug
            # 'mlp_mixer',  # bug
            # 'conformer',  # bug
            # 'poolformer',  # bug
            # 'vit',  # bug
            # 'efficientformer',
            # 'mobileone',
            # 'edgenext'
        ]
        mmcls_exclude = ['cutmix', 'cifar', 'gem']
        mmcls_model_library = MMClsModelLibrary(
            include=mmcls_model_include, exclude=mmcls_exclude)
        return mmcls_model_library

    @classmethod
    def mmdet_library(cls):
        mmdet_include = [
            # 'rpn',  #
            # 'faster-rcnn',
            # 'cascade-rcnn',
            # 'fast-rcnn',  # mmdet has bug.
            # 'retinanet',
            # 'mask-rcnn',
            # 'ssd300'
        ]
        mmdet_model_library = MMDetModelLibrary(mmdet_include)
        return mmdet_model_library

    @classmethod
    def mmseg_library(cls):
        include = [
            # 'cgnet',
            # 'gcnet',
            # 'setr',
            # 'deeplabv3',
            # 'twins',
            # 'fastfcn',
            # 'fpn',
            # 'upernet',
            # 'dnl',
            # 'icnet',
            # 'segmenter',
            # 'encnet',
            # 'erfnet',
            # 'segformer',
            # 'apcnet',
            # 'fast',
            # 'ocrnet',
            # 'lraspp',
            # 'dpt',
            # 'fcn',
            # 'psanet',
            # 'bisenetv2',
            # 'pointrend',
            # 'ccnet',
            'pspnet',
            # 'dmnet',
            # 'stdc',
            # 'ann',
            # 'nonlocal',
            # 'isanet',
            # 'danet',
            # 'emanet',
            # 'deeplabv3plus',
            # 'bisenetv1',
        ]
        model_library = MMSegModelLibrary(include=include)
        return model_library

    @classmethod
    def librarys(cls) -> List[ModelLibrary]:
        return [
            cls.default_library(),
            cls.torch_library(),
            cls.mmcls_library(),
            cls.mmseg_library(),
            cls.mmdet_library(),
        ]

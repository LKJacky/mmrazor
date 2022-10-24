# Copyright (c) OpenMMLab. All rights reserved.
from .model_library import (MMClsModelLibrary, MMDetModelLibrary, ModelLibrary,
                            DefaultModelLibrary, TorchModelLibrary,
                            MMSegModelLibrary)


class PassedModelManager:

    def __init__(self) -> None:
        pass

    def include_models(self, full_test=False):
        models = []
        for library in self.libraries(full_test):
            models.extend(library.include_models())
        return models

    def uninclude_models(self, full_test=False):
        models = []
        for library in self.libraries(full_test):
            models.extend(library.uninclude_models())
        return models

    def libraries(self, full=False):
        return []


class FxPassedModelManager(PassedModelManager):

    _default_library = None
    _torch_library = None
    _mmcls_library = None
    _mmseg_library = None
    _mmdet_library = None

    def libraries(self, full=False):
        if full:
            return [
                self.__class__.default_library(),
                self.__class__.torch_library(),
                self.__class__.mmcls_library(),
                self.__class__.mmseg_library(),
                self.__class__.mmdet_library(),
            ]
        else:
            return [self.__class__.default_library()]

    @classmethod
    def default_library(cls):
        if cls._default_library is None:
            cls._default_library = DefaultModelLibrary(include=[
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

        return cls._default_library

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
        if cls._torch_library is None:
            cls._torch_library = TorchModelLibrary(include=torch_includes)
        return cls._torch_library

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
        if cls._mmcls_library is None:
            cls._mmcls_library = MMClsModelLibrary(include=mmcls_include)
        return cls._mmcls_library

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
        if cls._mmdet_library is None:
            cls._mmdet_library = MMDetModelLibrary(mmdet_include)
        return cls._mmdet_library

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
        if cls._mmseg_library is None:
            cls._mmseg_library = MMSegModelLibrary(include=include)
        return cls._mmseg_library

    # for backward tracer


class BackwardPassedModelManager(PassedModelManager):

    _default_library = None
    _torch_library = None
    _mmcls_library = None
    _mmseg_library = None
    _mmdet_library = None

    def libraries(self, full=False):
        if full:
            return [
                self.__class__.default_library(),
                self.__class__.torch_library(),
                self.__class__.mmcls_library(),
                self.__class__.mmseg_library(),
                self.__class__.mmdet_library(),
            ]
        else:
            return [self.__class__.default_library()]

    @classmethod
    def default_library(cls):
        if cls._default_library is None:
            cls._default_library = DefaultModelLibrary(include=[
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
        return cls._default_library

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
        if cls._torch_library is None:
            cls._torch_library = TorchModelLibrary(include=torch_includes)
        return cls._torch_library

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
        if cls._mmcls_library is None:
            cls._mmcls_library = MMClsModelLibrary(
                include=mmcls_model_include, exclude=mmcls_exclude)
        return cls._mmcls_library

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
        if cls._mmdet_library is None:
            cls._mmdet_library = MMDetModelLibrary(mmdet_include)
        return cls._mmdet_library

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
        if cls._mmseg_library is None:
            cls._mmseg_library = MMSegModelLibrary(include=include)
        return cls._mmseg_library


backward_passed_library = FxPassedModelManager()
fx_passed_library = BackwardPassedModelManager()

# Copyright (c) OpenMMLab. All rights reserved.
from .model_library import (MMClsModelLibrary, MMDetModelLibrary,
                            TorchModelLibrary, MMSegModelLibrary)
from .models import Icep  # noqa
from .models import MultipleUseModel  # noqa
from .models import Xmodel  # noqa
from .models import (AddCatModel, ConcatModel, ConvAttnModel, DwConvModel,
                     ExpandLineModel, GroupWiseConvModel, LineModel,
                     MultiBindModel, MultiConcatModel, MultiConcatModel2,
                     ResBlock)
import os

FULL_TEST = os.getenv('FULL_TEST') == 'true'


class PassedModelManager:

    # for fx tracer

    @classmethod
    def fx_tracer_passed_default_models(cls):
        default_models = [
            LineModel,
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
        ]
        return default_models

    @classmethod
    def fx_tracer_passed_torch_models(cls):
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
        return torch_model_library.include_models()

    @classmethod
    def fx_tracer_passed_mmcls_models(cls):
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
        mmcls_exclude = ['cutmix', 'cifar', 'gem']
        mmcls_model_library = MMClsModelLibrary(
            include=mmcls_include, exclude=mmcls_exclude)
        return mmcls_model_library.include_models()

    @classmethod
    def fx_tracer_passed_mmdet_models(cls):
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
        return mmdet_model_library.include_models()

    @classmethod
    def fx_tracer_passed_mmseg_models(cls):
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
        return model_library.include_models()

    @classmethod
    def fx_tracer_passed_models(cls):


        models = cls.fx_tracer_passed_default_models() \
            + cls.fx_tracer_passed_torch_models() \
            + cls.fx_tracer_passed_mmcls_models() \
            + cls.fx_tracer_passed_mmdet_models() \
            + cls.fx_tracer_passed_mmseg_models() \
            if FULL_TEST else cls.fx_tracer_passed_default_models()

        return models

    # for backward tracer

    @classmethod
    def backward_tracer_passed_default_models(cls):
        '''MultipleUseModel: backward tracer can't distinguish multiple use and
        first bind then use.'''
        default_models = [
            LineModel,
            ResBlock,
            AddCatModel,
            ConcatModel,
            MultiConcatModel,
            MultiConcatModel2,
            GroupWiseConvModel,
            Xmodel,
            # MultipleUseModel,  # bug
            Icep,
            ExpandLineModel,
            MultiBindModel,
            DwConvModel
        ]
        return default_models

    @classmethod
    def backward_tracer_passed_torch_models(cls):
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
        return torch_model_library.include_models()

    @classmethod
    def backward_tracer_passed_mmcls_models(cls):
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
        return mmcls_model_library.include_models()

    @classmethod
    def backward_tracer_passed_mmdet_models(cls):
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
        return mmdet_model_library.include_models()

    @classmethod
    def backward_tracer_passed_mmseg_models(cls):
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
        return model_library.include_models()

    @classmethod
    def backward_tracer_passed_models(cls):
        models = cls.backward_tracer_passed_default_models() \
            + cls.backward_tracer_passed_torch_models() \
            + cls.backward_tracer_passed_mmcls_models() \
            + cls.backward_tracer_passed_mmdet_models() \
            + cls.backward_tracer_passed_mmseg_models() \
            if FULL_TEST else cls.backward_tracer_passed_default_models()

        return models
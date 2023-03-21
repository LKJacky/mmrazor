# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from .op import DynamicBlockMixin


class ResLayer(nn.Sequential):
    pass


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BaseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BaseBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """For CIFAR10 ResNet paper uses option A."""
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4), 'constant', 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(BaseBasicBlock, DynamicBlockMixin):
    expansion = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dynamic_block_init()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) * self.scale
        out = F.relu(out)
        return out

    @property
    def is_removable(self):
        return len(self.shortcut) == 0

    def to_static_op(self) -> nn.Module:
        from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
        module = BaseBasicBlock(*self.args, **self.kwargs)
        for name, m in self.named_children():
            assert hasattr(module, name)
            setattr(module, name, _dynamic_to_static(m))
        return module

    @property
    def out_channel(self):
        return self.conv2.out_channels


@MODELS.register_module()
class ResNetCifarDMS(nn.Module):

    def __init__(self, num_blocks=[9, 9, 9], num_classes=10):
        block = BasicBlock
        super(ResNetCifarDMS, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return ResLayer(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return (out, )


@MODELS.register_module()
def ResNetCifarSuper(ratio=2.0,
                     num_blocks=[12, 12, 12],
                     init_cfg=None,
                     data_preprocessor=None,
                     *args,
                     **kwargs):
    from mmcls.models.classifiers import ImageClassifier
    model: ImageClassifier = MODELS.build(
        dict(
            type='mmcls.ImageClassifier',
            backbone=dict(
                type='mmrazor.ResNetCifarDMS',
                num_blocks=num_blocks,
                num_classes=10),
            head=dict(
                type='mmcls.LinearClsHead',
                num_classes=10,
                in_channels=64,
                loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            ),
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor))
    from mmrazor.models.utils.expandable_utils import (
        expand_expandable_dynamic_model, to_expandable_model)
    mutator = to_expandable_model(model)
    for unit in mutator.mutable_units:
        unit.expand_to(int(unit.current_choice * ratio))

    model = expand_expandable_dynamic_model(model)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.reset_parameters()
        elif isinstance(module, nn.BatchNorm2d):
            module.reset_parameters()
        elif isinstance(module, nn.Linear):
            module.reset_parameters()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            from mmrazor.utils import print_log
            print_log(f'{type(module)} is not initialized')

    print_log(f'{kwargs.keys()} are not used in Super ResNetCifar')
    print_log(f'Super ResNetCifar {model}')
    return model


if __name__ == '__main__':
    model = ResNetCifarDMS()
    MODEL = dict(
        type='mmcls.ImageClassifier',
        backbone=dict(type='mmrazor.ResNetCifar', num_classes=10),
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=10,
            in_channels=64,
            loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        ))
    import torch
    x = torch.randn(1, 3, 32, 32)
    model = MODELS.build(MODEL)
    y = model(x)
    print(y.shape)

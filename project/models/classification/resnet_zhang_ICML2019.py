from typing import Type, Callable, Union, List, Optional

import torch
import torch.nn as nn
from antialiased_cnns import BlurPool
from torch import Tensor

from .base_net import BaseNet


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            antialiasing_filter_size=1
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)  # Conv(stride2)-Norm-Relu --> #Conv-Norm-Relu-BlurPool(stride2)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = nn.Sequential(BlurPool(planes, filt_size=antialiasing_filter_size, stride=stride),
                                       conv3x3(planes, planes), )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            antialiasing_filter_size=1,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, groups=groups,
                             dilation=dilation)  # Conv(stride2)-Norm-Relu --> #Conv-Norm-Relu-BlurPool(stride2)
        self.bn2 = norm_layer(width)
        if stride == 1:
            self.conv3 = conv1x1(width, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(BlurPool(width, filt_size=antialiasing_filter_size, stride=stride),
                                       conv1x1(width, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(BaseNet):
    def __init__(
            self,
            args, *,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            antialiasing_filter_size: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Sequential(*[BlurPool(self.in_planes, filt_size=antialiasing_filter_size, stride=2, ),
                                       nn.MaxPool2d(kernel_size=2, stride=1),
                                       BlurPool(self.in_planes, filt_size=antialiasing_filter_size, stride=2, )])

        if args.dataset_name == 'cifar10':  # input image resolution is 32x32
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1)
            self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       antialiasing_filter_size=antialiasing_filter_size)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       antialiasing_filter_size=antialiasing_filter_size)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       antialiasing_filter_size=antialiasing_filter_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, self.hparams.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            antialiasing_filter_size=1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            # since this is just a conv1x1 layer (no nonlinearity),
            # conv1x1->blurpool is the same as blurpool->conv1x1; the latter is cheaper
            downsample = [BlurPool(filt_size=antialiasing_filter_size, stride=stride, channels=self.in_planes), ] if (
                    stride != 1) else []
            downsample += [conv1x1(self.in_planes, planes * block.expansion, 1),
                           norm_layer(planes * block.expansion)]
            downsample = nn.Sequential(*downsample)

        layers = [block(
            self.in_planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
            antialiasing_filter_size
        )]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    antialiasing_filter_size=antialiasing_filter_size,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(self.relu(self.bn1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def resnet18(args, antialiasing_filter_size=4) -> ResNet:
    return ResNet(args, block=BasicBlock, layers=[2, 2, 2, 2], antialiasing_filter_size=antialiasing_filter_size)


def resnet34(args, antialiasing_filter_size=4) -> ResNet:
    return ResNet(args, block=BasicBlock, layers=[3, 4, 6, 3], antialiasing_filter_size=antialiasing_filter_size)


def resnet50(args, antialiasing_filter_size=4) -> ResNet:
    return ResNet(args, block=Bottleneck, layers=[3, 4, 6, 3], antialiasing_filter_size=antialiasing_filter_size)


def resnet101(args, antialiasing_filter_size=4) -> ResNet:
    return ResNet(args, block=Bottleneck, layers=[3, 4, 23, 3], antialiasing_filter_size=antialiasing_filter_size)


def resnet152(args, antialiasing_filter_size=4) -> ResNet:
    return ResNet(args, block=Bottleneck, layers=[3, 8, 36, 3], antialiasing_filter_size=antialiasing_filter_size)


def resnext50_32x4d(args, antialiasing_filter_size=4, **kwargs) -> ResNet:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNet(args, block=Bottleneck, layers=[3, 4, 6, 3], antialiasing_filter_size=antialiasing_filter_size,
                  **kwargs)


def resnext101_32x8d(args, antialiasing_filter_size=4, **kwargs) -> ResNet:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return ResNet(args, block=Bottleneck, layers=[3, 4, 23, 3], antialiasing_filter_size=antialiasing_filter_size,
                  **kwargs)


def resnext101_64x4d(args, antialiasing_filter_size=4, **kwargs) -> ResNet:
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return ResNet(args, block=Bottleneck, layers=[3, 4, 23, 3], antialiasing_filter_size=antialiasing_filter_size,
                  **kwargs)

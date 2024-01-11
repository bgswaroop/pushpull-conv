import math
from typing import Type, Callable, Union, List, Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .base_net import BaseNet


class PPmodule2dStrisciuglioEtAl(nn.Module):
    """
    Implementation of the Push-Pull layer from:
    [1] N. Strisciuglio, M. Lopez-Antequera, N. Petkov,
    Enhanced robustness of convolutional networks with a push–pull inhibition layer,
    Neural Computing and Applications, 2020, doi: 10.1007/s00521-020-04751-8

    It is an extension of the Conv2d module, with extra arguments:

    * :attr:`alpha` controls the weight of the inhibition. (default: 1 - same strength as the push kernel)
    * :attr:`scale` controls the size of the pull (inhibition) kernel (default: 2 - double size).
    * :attr:`dual_output` determines if the response maps are separated for push and pull components.
    * :attr:`train_alpha` controls if the inhibition strength :attr:`alpha` is trained (default: False).


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        alpha (float, optional): Strength of the inhibitory (pull) response. Default: 1
        scale (float, optional): size factor of the pull (inhibition) kernel with respect to the pull kernel. Default: 2
        dual_output (bool, optional): If ``True``, push and pull response maps are places into separate channels of the output. Default: ``False``
        train_alpha (bool, optional): If ``True``, set alpha (inhibition strength) as a learnable parameters. Default: ``False``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 alpha=1, scale=2, dual_output=False,
                 train_alpha=False):
        super(PPmodule2dStrisciuglioEtAl, self).__init__()

        self.dual_output = dual_output
        self.train_alpha = train_alpha

        # Note: the dual output is not tested yet
        if self.dual_output:
            assert (out_channels % 2 == 0)
            out_channels = out_channels // 2

        # Push kernels (is the one for which the weights are learned - the pull kernel is derived from it)
        self.push = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

        """
        # Bias: push and pull convolutions will have bias=0.
        # If the PP kernel has bias, it is computed next to the combination of the 2 convolutions
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            # Initialize bias
            n = in_channels
            for k in self.push.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        """

        # Configuration of the Push-Pull inhibition
        if not self.train_alpha:
            # when alpha is an hyper-parameter (as in [1])
            self.alpha = alpha
        else:
            # when alpha is a trainable parameter
            k = 1
            self.alpha = nn.Parameter(k * torch.ones(1, out_channels, 1, 1), requires_grad=True)
            r = 1. / math.sqrt(in_channels * out_channels)
            self.alpha.data.uniform_(.5 - r, .5 + r)  # math.sqrt(n) / 2)  # (-stdv, stdv)

        self.scale_factor = scale
        push_size = self.push.weight[0].size()[1]

        # compute the size of the pull kernel
        if self.scale_factor == 1:
            pull_size = push_size
        else:
            pull_size = math.floor(push_size * self.scale_factor)
            if pull_size % 2 == 0:
                pull_size += 1

        # up-sample the pull kernel from the push kernel
        self.pull_padding = pull_size // 2 - push_size // 2 + padding
        self.up_sampler = nn.Upsample(size=(pull_size, pull_size),
                                      mode='bilinear',
                                      align_corners=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # with torch.no_grad():
        if self.scale_factor == 1:
            pull_weights = self.push.weight
        else:
            pull_weights = self.up_sampler(self.push.weight)
        # pull_weights.requires_grad = False

        bias = self.push.bias
        if self.push.bias is not None:
            bias = -self.push.bias

        push = self.relu(self.push(x))
        pull = self.relu(F.conv2d(x,
                                  -pull_weights,
                                  bias,
                                  self.push.stride,
                                  self.pull_padding, self.push.dilation,
                                  self.push.groups))

        alpha = self.alpha
        if self.train_alpha:
            # alpha is greater or equal than 0
            alpha = self.relu(self.alpha)

        if self.dual_output:
            x = torch.cat([push, pull], dim=1)
        else:
            x = push - alpha * pull
            # + self.bias.reshape(1, self.push.out_channels, 1, 1) #.repeat(s[0], 1, s[2], s[3])
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


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
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
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

        if args.use_push_pull and args.num_push_pull_layers >= 1:
            self.conv1 = PPmodule2dStrisciuglioEtAl(in_channels=3, out_channels=self.in_planes,
                                                    kernel_size=(7, 7), stride=2, padding=3, bias=True,
                                                    alpha=args.pull_inhibition_strength,
                                                    train_alpha=args.trainable_pull_inhibition)
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.bn = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if args.dataset_name == 'cifar10':
            if args.use_push_pull and args.num_push_pull_layers >= 1:
                self.conv1 = PPmodule2dStrisciuglioEtAl(in_channels=3, out_channels=self.in_planes,
                                                        kernel_size=(3, 3), stride=1, padding=1, bias=True,
                                                        alpha=args.pull_inhibition_strength,
                                                        train_alpha=args.trainable_pull_inhibition)
            else:
                self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1)
            self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
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
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(
            self.in_planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
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
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(self.relu(self.bn(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def resnet18(args) -> ResNet:
    return ResNet(args, block=BasicBlock, layers=[2, 2, 2, 2])


def resnet34(args) -> ResNet:
    return ResNet(args, block=BasicBlock, layers=[3, 4, 6, 3])


def resnet50(args) -> ResNet:
    return ResNet(args, block=Bottleneck, layers=[3, 4, 6, 3])


def resnet101(args) -> ResNet:
    return ResNet(args, block=Bottleneck, layers=[3, 4, 23, 3])


def resnet152(args) -> ResNet:
    return ResNet(args, block=Bottleneck, layers=[3, 8, 36, 3])


def resnext50_32x4d(args, **kwargs) -> ResNet:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNet(args, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)


def resnext101_32x8d(args, **kwargs) -> ResNet:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return ResNet(args, block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)


def resnext101_64x4d(args, **kwargs) -> ResNet:
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return ResNet(args, block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)

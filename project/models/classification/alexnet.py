import torch
from torch import nn as nn

from .base_net import BaseNet
from ..utils.push_pull_unit import PushPullConv2DUnit


class AlexNet(BaseNet):
    def __init__(self, args, dropout: float = 0.5, ):
        super(AlexNet, self).__init__()
        self.save_hyperparameters(args)

        if args.dataset_name in {'imagenet', 'imagenet100', 'imagenet200'}:
            self.features = self._alexnet_for_imagenet(args)
        elif args.dataset_name == 'cifar10':
            self.features = self._alexnet_for_cifar(args)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.hparams.num_classes),
        )

    @staticmethod
    def _alexnet_for_imagenet(args):
        """
        The original implementation of the AlexNet is designed to handle inputs of size 256x256
        along with PushPull inhibition
        :param args:
        :return:
        """
        in_channels = 1 if args.use_grayscale else 3
        if args.use_push_pull and args.num_push_pull_layers >= 1:
            conv1 = PushPullConv2DUnit(in_channels, out_channels=64,
                                       kernel_size=(11, 11),
                                       avg_kernel_size=args.avg_kernel_size,
                                       pull_inhibition_strength=args.pull_inhibition_strength,
                                       trainable_pull_inhibition=args.trainable_pull_inhibition,
                                       stride=4,
                                       padding=2)
        else:
            conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=(11, 11), stride=4, padding=2)

        features = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        return features

    @staticmethod
    def _alexnet_for_cifar(args):
        """
        This is a trimmed down version to handle input images of size 32x32
        The original implementation of the AlexNet is designed to handle inputs of size 256x256
        :param args:
        :return:
        """
        in_channels = 1 if args.use_grayscale else 3
        if args.use_push_pull and args.num_push_pull_layers >= 1:
            conv1 = PushPullConv2DUnit(in_channels, out_channels=64,
                                       kernel_size=(3, 3),
                                       avg_kernel_size=args.avg_kernel_size,
                                       pull_inhibition_strength=args.pull_inhibition_strength,
                                       trainable_pull_inhibition=args.trainable_pull_inhibition,
                                       stride=2,
                                       padding=1)
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)

        features = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

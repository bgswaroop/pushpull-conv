import torch
from torch import nn as nn

from .base_net import BaseNet
from ..utils.push_pull_unit import PushPullConv2DUnit


class AlexNet(BaseNet):
    def __init__(self, args, dropout: float = 0.5, ):
        super(AlexNet, self).__init__()
        self.save_hyperparameters(args)
        if args.use_push_pull and args.num_push_pull_layers >= 1:
            conv1 = PushPullConv2DUnit(in_channels=3, out_channels=64,
                                       push_kernel_size=args.push_kernel_size,
                                       pull_kernel_size=args.pull_kernel_size,
                                       avg_kernel_size=args.avg_kernel_size,
                                       pupu_weight=args.pupu_weight,
                                       pull_inhibition_strength=args.pull_inhibition_strength,
                                       stride=4,
                                       padding=2)
        else:
            conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11), stride=4, padding=2)

        self.features = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
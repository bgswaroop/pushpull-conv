import torch

from .base_net import BaseNet
from ..utils.push_pull_unit import PushPullConv2DUnit


class ConvNet(BaseNet):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        if args.use_push_pull and args.num_push_pull_layers >= 1:
            self.conv1 = PushPullConv2DUnit(in_channels=3, out_channels=32,
                                            push_kernel_size=args.push_kernel_size,
                                            pull_kernel_size=args.pull_kernel_size,
                                            avg_kernel_size=args.avg_kernel_size,
                                            pupu_weight=args.pupu_weight,
                                            pull_inhibition_strength=args.pull_inhibition_strength,
                                            padding='same')
        else:
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding='same')

        if args.use_push_pull and args.num_push_pull_layers >= 2:
            self.conv2 = PushPullConv2DUnit(in_channels=32, out_channels=32,
                                            push_kernel_size=args.push_kernel_size,
                                            pull_kernel_size=args.pull_kernel_size,
                                            avg_kernel_size=args.avg_kernel_size,
                                            pupu_weight=args.pupu_weight,
                                            pull_inhibition_strength=args.pull_inhibition_strength,
                                            padding='same')
        else:
            self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding='same')

        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding='same')
        self.fc1 = torch.nn.Linear(in_features=3 * 3 * 64, out_features=500)
        self.fc2 = torch.nn.Linear(in_features=500, out_features=self.hparams.num_classes)

        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            torch.nn.init.xavier_uniform_(layer.weight)

        self.maxPool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

    def forward(self, x):
        x = torch.relu(self.maxPool(self.conv1(x)))
        x = torch.relu(self.maxPool(self.conv2(x)))
        x = torch.relu(self.maxPool(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
from typing import Union

import torch
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair


class PushPullConv2DUnit(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            # kernel_size: _size_2_t,
            push_kernel_size: _size_2_t,
            pull_kernel_size: _size_2_t,
            avg_kernel_size: _size_2_t,
            pull_inhibition_strength: int = 1,
            scale_the_outputs: bool = False,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None):
        super(PushPullConv2DUnit, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.pull_inhibition_strength = pull_inhibition_strength
        self.scale_the_outputs = scale_the_outputs
        self.pull_kernel_size = pull_kernel_size

        # self.pull_inhibition_strength = torch.nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
        # self.pull_inhibition_strength.data.uniform_(0, 1)

        self.push_conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=push_kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, device=device,
            dtype=dtype)

        if avg_kernel_size != 0:
            self.avg = torch.nn.AvgPool2d(
                kernel_size=avg_kernel_size,
                stride=1,
                padding=tuple([int((x - 1) / 2) for x in _pair(avg_kernel_size)])
            )
        else:
            self.avg = None

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            self.bias.data.uniform_(-1, 1)  # random weight initialization
        else:
            self.bias = None

    @property
    def weight(self):
        return self.push_conv.weight

    @weight.setter
    def weight(self, value):
        self.push_conv.weight = value

    def forward(self, x):
        push_response = F.relu_(self.push_conv(x))
        # pull_conv_kernel = F.interpolate(-self.push_conv.weight, size=self.pull_kernel_size, mode='bilinear')
        pull_response = F.relu_(F.conv2d(input=x, weight=-self.push_conv.weight, stride=self.stride,
                                         padding=self.padding, dilation=self.dilation, groups=self.groups))

        if self.avg:
            pull_response = self.avg(pull_response)
        # else:
            # avg_pull_response = None
            # x_out = F.relu_(push_response - self.pull_inhibition_strength.view((1, -1, 1, 1)) * pull_response)
        x_out = F.relu_(push_response - self.pull_inhibition_strength * pull_response)

        # if self.scale_the_outputs:
        #     ratio = torch.amax(push_response, dim=(2, 3), keepdim=True) / \
        #             (torch.amax(x_out, dim=(2, 3), keepdim=True) + 1e-20)
        #     x_out = x_out * ratio

        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))

        # plot_push_pull_kernels(push_response, pull_response, avg_pull_response, x, x_out, x_out_scaled, k=0)
        return x_out
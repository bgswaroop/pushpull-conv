from typing import Union

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair


class PushPullConv2DUnit(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            push_kernel_size: _size_2_t,
            pull_kernel_size: _size_2_t,
            avg_kernel_size: _size_2_t,
            pull_inhibition_strength: int = 1,
            trainable_pull_inhibition: bool = False,
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
        self.pull_kernel_size = pull_kernel_size

        if trainable_pull_inhibition:
            self.pull_inhibition_strength = torch.nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            self.pull_inhibition_strength.data.uniform_(0, 1)
        else:
            self.pull_inhibition_strength = torch.tensor([pull_inhibition_strength])

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
            x_out = F.relu_(push_response - pull_response * self.pull_inhibition_strength)
        else:
            x_out = F.relu_(push_response - pull_response * self.pull_inhibition_strength.view((1, -1, 1, 1)))


        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))

        # plot_push_pull_kernels(push_response, pull_response, avg_pull_response, x, x_out, x_out_scaled, k=0)
        return x_out


def plot_push_pull_kernels(push_response, pull_response, avg_pull_response, x, x_out, x_out_scaled, k=0):
    fig, ax = plt.subplots(2, 3)
    ax1 = ax[0][0]
    ax2 = ax[0][1]
    ax3 = ax[1][0]
    ax4 = ax[1][1]
    ax5 = ax[0][2]
    ax6 = ax[1][2]

    im1 = ax1.imshow(push_response[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    ax1.set_title('push_response')

    im2 = ax2.imshow(x_out[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    ax2.set_title('final_response')

    im3 = ax3.imshow(pull_response[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')
    ax3.set_title('pull_response')

    im4 = ax4.imshow(avg_pull_response[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im4, cax=cax, orientation='vertical')
    ax4.set_title('avg_pull_response')

    im5 = ax5.imshow(x[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im5, cax=cax, orientation='vertical')
    ax5.set_title('input')

    im6 = ax6.imshow(x_out_scaled[0, k, :, :].cpu().detach().numpy())
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im6, cax=cax, orientation='vertical')
    ax6.set_title('final_response_scaled')

    plt.tight_layout()
    plt.show()
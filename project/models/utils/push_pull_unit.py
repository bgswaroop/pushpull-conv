import math
from typing import Union

import numpy as np
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
            kernel_size: _size_2_t,
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
        self.trainable_pull_inhibition = trainable_pull_inhibition

        if trainable_pull_inhibition:
            self.pull_inhibition_strength = torch.nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            self.pull_inhibition_strength.data.uniform_(0, 1)
        else:
            self.pull_inhibition_strength = pull_inhibition_strength

        self.push_conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
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
        # plot_data = [('input', x)]
        W = self.push_conv.weight
        min_push = torch.amin(W, dim=(1, 2, 3), keepdim=True)
        max_push = torch.amax(W, dim=(1, 2, 3), keepdim=True)
        pull_kernel = -W + (max_push + min_push)

        push_response = self.push_conv(x)
        pull_response = F.conv2d(x, pull_kernel, None, self.stride, self.padding, self.dilation, self.groups)
        if self.avg:
            pull_response = self.avg(pull_response)
        # plot_data.extend([('push_response', push_response), ('pull_response', pull_response)])

        push_response = F.relu_(push_response)
        pull_response = F.relu_(pull_response)
        # plot_data.extend([('push_response + ReLU', push_response), ('pull_response + ReLU', pull_response)])

        if not self.trainable_pull_inhibition:
            x_out = push_response - pull_response * self.pull_inhibition_strength
        else:
            x_out = push_response - pull_response * self.pull_inhibition_strength.view((1, -1, 1, 1))
        # plot_data.extend([('x_out', x_out)])

        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))
        # plot_data.extend([('x_out + bias', x_out)])

        # plot_minibatch_inputs(x)
        # plot_push_kernels(self.push_conv.weight)
        # plot_intermediate_response(plot_data, img_index=0, filters_to_plot=(9,))

        return x_out


def plot_intermediate_response(plot_data, img_index=0, filters_to_plot=(0,)):
    # bring all tensors from GPU to CPU
    plot_data_cpu = [(name, tensor[img_index].cpu().detach().numpy()) for name, tensor in plot_data]

    # plot attributes
    num_rows = 2
    num_cols = math.ceil(len(plot_data_cpu) / num_rows)

    for filter_id in filters_to_plot:
        fig, ax = plt.subplots(num_rows, num_cols, dpi=200, figsize=(num_cols * 2, num_rows * 2))
        row_id, col_id = 0, 0
        for name, tensor in plot_data_cpu:
            if name == 'input':
                arr_to_plot = np.transpose(tensor, axes=[1, 2, 0])  # channels last
            else:
                arr_to_plot = tensor[filter_id, :, :]
            img = ax[row_id][col_id].imshow(arr_to_plot)

            divider = make_axes_locatable(ax[row_id][col_id])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(img, cax=cax, orientation='vertical')
            ax[row_id][col_id].set_title(name)
            ax[row_id][col_id].axis('off')

            col_id = (col_id + 1) % num_cols
            if col_id == 0:
                row_id += 1

        plt.tight_layout()
        plt.show()
        plt.close()


def plot_minibatch_inputs(plot_data):
    plot_data_cpu = [tensor.cpu().detach().numpy() for tensor in plot_data]
    num_rows = num_cols = math.ceil(math.sqrt(len(plot_data_cpu)))
    fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, dpi=200, figsize=(15, 15))
    row_id, col_id = 0, 0

    for idx, tensor in enumerate(plot_data_cpu):
        arr_to_plot = np.transpose(tensor, axes=[1, 2, 0])  # channels last
        ax[row_id][col_id].imshow(arr_to_plot)
        ax[row_id][col_id].set_title(f'{idx}')

        col_id = (col_id + 1) % num_cols
        if col_id == 0:
            row_id += 1

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_push_kernels(plot_data):
    plot_data_cpu = [tensor.cpu().detach().numpy() for tensor in plot_data]
    num_rows = num_cols = math.ceil(math.sqrt(len(plot_data_cpu)))
    fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(15, 7), constrained_layout=True)
    row_id, col_id = 0, 0
    for idx, tensor in enumerate(plot_data_cpu):
        # arr_to_plot = np.transpose(tensor, axes=[1, 2, 0])  # channels last
        img = ax[row_id][col_id].imshow(np.concatenate([tensor[0], tensor[1], tensor[2]], axis=1))
        divider = make_axes_locatable(ax[row_id][col_id])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img, cax=cax, orientation='vertical')
        ax[row_id][col_id].set_title(f'{idx}')
        col_id = (col_id + 1) % num_cols
        if col_id == 0:
            row_id += 1
    plt.tight_layout()
    plt.show()
    plt.close()


import torch
import math
from torch import nn
import torch.nn.functional as F


class PPmodule2d(nn.Module):
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
        super(PPmodule2d, self).__init__()

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
            # Inizialize bias
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
            self.alpha.data.uniform_(.5-r, .5+r)  # math.sqrt(n) / 2)  # (-stdv, stdv)

        self.scale_factor = scale
        push_size = self.push.weight[0].size()[1]

        # compute the size of the pull kernel
        if self.scale_factor == 1:
            pull_size = push_size
        else:
            pull_size = math.floor(push_size * self.scale_factor)
            if pull_size % 2 == 0:
                pull_size += 1

        # upsample the pull kernel from the push kernel
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

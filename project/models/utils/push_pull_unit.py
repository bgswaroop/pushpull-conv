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
            push_kernel_size: _size_2_t,
            pull_kernel_size: _size_2_t,
            avg_kernel_size: _size_2_t,
            pull_inhibition_strength: int = 1,
            trainable_pull_inhibition: bool = False,
            pupu_weight: float = 1.0,
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
        self.plot_kernels = False  # debug flag
        self.skip_connection = True
        self.pupu_weight = pupu_weight  # controls the pull inhibition

        if trainable_pull_inhibition:
            self.pull_inhibition_strength = torch.nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            self.pull_inhibition_strength.data.uniform_(0, 1)
        else:
            self.pull_inhibition_strength = pull_inhibition_strength

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

    # def forward(self, x):
    #     push_response = F.relu_(self.push_conv(x))
    #     pull_response = F.relu_(F.conv2d(input=x, weight=-self.push_conv.weight, stride=self.stride,
    #                                      padding=self.padding, dilation=self.dilation, groups=self.groups))
    #
    #     if self.avg:
    #         pull_response = self.avg(pull_response)
    #         x_out = F.relu_(push_response - pull_response * self.pull_inhibition_strength)
    #     else:
    #         x_out = F.relu_(push_response - pull_response * self.pull_inhibition_strength.view((1, -1, 1, 1)))
    #
    #     if self.bias is not None:
    #         x_out = x_out + self.bias.view((1, -1, 1, 1))
    #
    #     return x_out

    # Debug version of forward
    def forward(self, x):

        # plot_data = [('input', x)]
        push_response_wo_relu = self.push_conv(x)
        # plot_data.extend([('push_response', push_response_wo_relu)])
        push_response = F.relu_(push_response_wo_relu)
        pull_response = F.relu_(-push_response_wo_relu)
        # plot_data.extend([('push_response + ReLU', push_response), ('pull_response + ReLU', pull_response)])
        if self.avg:
            pull_response = self.avg(pull_response)
            x_out = F.relu_(push_response - pull_response * self.pull_inhibition_strength)
        else:
            x_out = F.relu_(push_response - pull_response * self.pull_inhibition_strength.view((1, -1, 1, 1)))
        # plot_data.extend([('avg_response', pull_response), ('PuPu output w/o bias', x_out)])

        # if self.skip_connection:
        #     # x_out = x_out + push_response_wo_relu  # skip connection
        #     x_out = self.pupu_weight * x_out + (1 - self.pupu_weight) * push_response_wo_relu  # weighted response
        # plot_data.extend([('After skip connection', x_out)])

        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))
        # plot_data.extend([('Final response', x_out)])

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

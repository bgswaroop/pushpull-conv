import math
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.fft import fft2, fftshift, ifftshift, ifft2
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair


# def gauss_kernel(l=5, sig=1.):
#     """\
#     credits: https://stackoverflow.com/a/43346070/2709971
#     creates gaussian kernel with side length `l` and a sigma of `sig`
#     """
#     ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
#     gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
#     kernel = np.outer(gauss, gauss)
#     return kernel / np.sum(kernel)

def surround_kernel(sigma=2):
    """
    The output kernel size is int(8 * sigma + 1)
    :param sigma:
    :return:
    """
    ax = np.linspace(int(-4 * sigma), int(4 * sigma), int(8 * sigma + 1))
    x, y = np.meshgrid(ax, ax)
    g1_sigma = 4 * sigma
    g2_sigma = sigma
    g1 = np.exp(-(x ** 2 + y ** 2) / (2 * g1_sigma ** 2)) / (2 * np.pi * g1_sigma ** 2)
    g2 = np.exp(-(x ** 2 + y ** 2) / (2 * g2_sigma ** 2)) / (2 * np.pi * g2_sigma ** 2)
    g = g1 - g2
    g[g < 0] = 0
    g = g / np.sum(g)
    return g


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
        self.out_channels = out_channels
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

        ss_kernel_2d = surround_kernel(sigma=2)
        ss_kernel = np.zeros((self.out_channels, self.out_channels, *ss_kernel_2d.shape))
        for index in range(ss_kernel_2d.shape[0]):
            ss_kernel[index, index] = ss_kernel_2d
        self.surround_kernel = torch.tensor(ss_kernel, device='cuda').to(torch.float32)

        if avg_kernel_size != 0:
            self.avg = torch.nn.AvgPool2d(
                kernel_size=avg_kernel_size,
                stride=1,
                padding=tuple([int((x - 1) / 2) for x in _pair(avg_kernel_size)]),
                count_include_pad=False
            )
        else:
            self.avg = None

        # pooling = gauss_kernel(kernel_size[0], sig=0.25)
        # pooling = pooling / np.max(pooling)
        # pooling = 1 - pooling
        # self.pooling = torch.tensor(pooling, device=device, dtype=dtype)

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
        """
        This is an implementation of surround-suppression. The idea is to inhibit the push response w.r.t its surround.
        This operation would suppress response to noise and fine-texture (for example, a single blade of grass).
        Features with high-contrast, especially high-level shapes would be preserved. Thereby introducing bias towards
        shape based features during the process of feature extraction. We believe, this bias would introduce robustness
        towards high-level shape based features.
        :param x: 4D input batch - (batch, channels, height, width)
        :return: surround-suppressed response of the convolution
        """

        # Compute the push-response
        push_response = self.push_conv(x)
        push_response = F.relu_(push_response)

        # Surround response
        surr_response = F.conv2d(push_response, self.surround_kernel, padding='same')

        # Suppressed push-response
        if not self.trainable_pull_inhibition:
            x_out = push_response - surr_response * self.pull_inhibition_strength
        else:
            x_out = push_response - surr_response * self.pull_inhibition_strength.view((1, -1, 1, 1))

        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))

        return x_out

    def forward_pushpull(self, x):
        """
        Zero-response to homogeneous patches
        :param x:
        :return:
        """
        push_kernel = self.push_conv.weight
        push_min = torch.amin(push_kernel, dim=(2, 3), keepdim=True)
        push_max = torch.amax(push_kernel, dim=(2, 3), keepdim=True)
        pull_kernel = -push_kernel + (push_max + push_min)
        push_sum = torch.sum(push_kernel, dim=(1, 2, 3), keepdims=True)
        pull_sum = torch.sum(pull_kernel, dim=(1, 2, 3), keepdims=True)

        eps = torch.finfo(torch.float32).eps
        pull_kernel = pull_kernel / (torch.abs(pull_sum) + eps) * (torch.abs(push_sum) + eps)
        inhibition_sign = torch.sign(push_sum) / torch.sign(pull_sum)

        push_response = self.push_conv(x)
        pull_response = F.conv2d(x, pull_kernel, None, self.stride, self.padding, self.dilation, self.groups)

        if self.avg:
            pull_response = self.avg(pull_response)
        # push_response = F.relu_(push_response)
        # pull_response = F.relu_(pull_response)

        if not self.trainable_pull_inhibition:
            x_out = push_response - pull_response * self.pull_inhibition_strength * inhibition_sign.view(1, -1, 1, 1)
        else:
            x_out = push_response - pull_response * self.pull_inhibition_strength.view((1, -1, 1, 1))

        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))

        return x_out

    def forward_dev(self, x):
        push_kernel = self.push_conv.weight
        min_push = torch.amin(push_kernel, dim=(2, 3), keepdim=True)
        max_push = torch.amax(push_kernel, dim=(2, 3), keepdim=True)
        pull_kernel = -push_kernel + (max_push + min_push)
        push_sum = torch.sum(push_kernel, dim=(2, 3), keepdims=True)
        pull_sum = torch.sum(pull_kernel, dim=(2, 3), keepdims=True)
        pull_kernel = pull_kernel / pull_sum * push_sum

        push_response = self.push_conv(x)
        pull_response = F.conv2d(x, pull_kernel, None, self.stride, self.padding, self.dilation, self.groups)

        if self.avg:
            pull_response = self.avg(pull_response)

        push_response = F.relu_(push_response)
        pull_response = F.relu_(pull_response)

        x_out = push_response - pull_response

        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))

        return x_out

    def _forward_cvpr(self, x):
        # plot_data = [('input', x)]
        W = self.push_conv.weight
        min_push = torch.amin(W, dim=(1, 2, 3), keepdim=True)
        max_push = torch.amax(W, dim=(1, 2, 3), keepdim=True)
        pull_kernel = -W + (max_push + min_push)
        # pull_kernel = self.get_pull_kernel(W, pull_kernel)

        # z = (W - W.mean(dim=(1, 2, 3), keepdims=True)) / W.std(dim=(1, 2, 3), keepdims=True)
        # max_std = 2
        # min_push = torch.amin(torch.where(torch.logical_and(z > -max_std, z < max_std), W, torch.inf),
        #                       dim=(1, 2, 3), keepdim=True)
        # max_push = torch.amax(torch.where(torch.logical_and(z > -max_std, z < max_std), W, -torch.inf),
        #                       dim=(1, 2, 3), keepdim=True)
        # pull_kernel = -W + (max_push + min_push)

        # push_sum = torch.abs(torch.sum(W, dim=(1, 2, 3), keepdims=True))
        # pull_sum = torch.abs(torch.sum(pull_kernel, dim=(1, 2, 3), keepdims=True))
        # pull_kernel = pull_kernel / pull_sum * push_sum
        # pull_kernel[:32] = self.normalize_pull_kernel(W[:32], pull_kernel[:32])

        push_response = self.push_conv(x)
        pull_response = F.conv2d(x, pull_kernel, None, self.stride, self.padding, self.dilation, self.groups)
        # pull_response = self.pull_conv(x)

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

    # def get_pull_kernel(self, push_kernel, pull_kernel):
    #     #     function pull_kernel = getPullKernel(push_kernel, sd)
    #     #     pull_kernel = -push_kernel + max(push_kernel(:)) + min(push_kernel(:));
    #     #     pooling = fspecial('gaussian', size(pull_kernel), sd);
    #     #     pooling = pooling / max(pooling(:));
    #     #     pooling = 1 - pooling;
    #     #     pull_kernel = ifft2(ifftshift(fftshift(fft2(pull_kernel)). * pooling));
    #
    #     d = (-2, -1)  # last two dimensions
    #     pull = fftshift(fft2(pull_kernel, dim=d), dim=d)
    #     pull = pull * self.pooling.to(pull.device)
    #     pull = torch.real(ifft2(ifftshift(pull, dim=d), dim=d)).float()
    #     return pull

    # def get_pull_kernel(self, push_kernel, pull_kernel):
    #     pos = push_kernel > 0
    #     neg = push_kernel < 0
    #     sumpos = torch.sum(torch.where(pos, push_kernel, 0), dim=(1, 2, 3))
    #     sumneg = torch.sum(torch.where(neg, push_kernel, 0), dim=(1, 2, 3), keepdim=True)
    #
    #     sumpos_or_sum_neg_is_zero = torch.logical_or(sumpos == 0.0, sumneg.view(-1) == 0.0)
    #     if torch.any(sumpos_or_sum_neg_is_zero):
    #         i = sumpos_or_sum_neg_is_zero
    #         push_sum = torch.sum(push_kernel, dim=(1, 2, 3), keepdims=True)
    #         pull_sum = torch.sum(pull_kernel, dim=(1, 2, 3), keepdims=True)
    #         pull_kernel[i] = pull_kernel[i] * push_sum[i] / pull_sum[i]
    #     else:
    #         sumpospull = torch.sum(torch.where(pull_kernel > 0, pull_kernel, 0), dim=(1, 2, 3))
    #         interval = 0.01 * (2 * (sumpospull < sumpos) - 1)
    #
    #         filter_ids = torch.argwhere(torch.logical_not(sumpos_or_sum_neg_is_zero)).view(-1)
    #         k = len(push_kernel[0].view(-1))  # number of weights in each filter
    #         s = push_kernel[0].shape  # kernel size
    #         d = push_kernel.device  # cpu or gpu device
    #
    #         for i in filter_ids:
    #             # determine the number of shifts to consider
    #             n = torch.abs(sumpos[i] - sumpospull[i]) / torch.abs(interval[i])
    #             if interval[i] < 0:
    #                 n = int(torch.ceil(torch.min(n, torch.max(pull_kernel[i]) / torch.abs(interval[i]))))
    #             else:
    #                 n = int(torch.abs(torch.floor(torch.min(n, torch.min(pull_kernel[i]) / torch.abs(interval[i])))))
    #
    #             # generate a matrix with all possible shifts
    #             pull = pull_kernel[i].view(-1).repeat(n, 1) + (torch.arange(n).to(d) * interval[i]).repeat(k, 1).T
    #             # sum the positive values in every row
    #             sumpospull_for_all_shifts = torch.sum(pull * (pull > 0), dim=1)
    #             # choose the row whose sum of positive values is the closest to the sum of all positive values in the push kernel
    #             min_idx = torch.argmin(torch.abs(sumpospull_for_all_shifts - sumpos[i]))
    #             # reshape the selected shifted kernel back to original dims
    #             pull_kernel[i] = pull[min_idx].view(s)
    #
    #         # normalize the negative values of the pull kernel to have the same values as that of the push
    #         pos_pull_values = torch.where(pull_kernel > 0, pull_kernel, 0)[filter_ids]
    #         neg_pull_values = torch.where(pull_kernel < 0, pull_kernel, 0)[filter_ids]
    #         sumpullneg = torch.sum(neg_pull_values, dim=(1, 2, 3), keepdim=True)
    #         pull_kernel[filter_ids] = (neg_pull_values * sumneg[filter_ids] / sumpullneg) + pos_pull_values
    #
    #     return pull_kernel

    # def get_pull_kernel(self, push_kernel, pull_kernel):
    #     pos = push_kernel > 0
    #     neg = push_kernel < 0
    #     sumpos = torch.sum(torch.where(pos, push_kernel, 0), dim=(1, 2, 3))
    #     sumneg = torch.sum(torch.where(neg, push_kernel, 0), dim=(1, 2, 3))
    #
    #     sumpos_or_sum_neg_is_zero = torch.logical_or(sumpos == 0.0, sumneg == 0.0)
    #     if torch.any(sumpos_or_sum_neg_is_zero):
    #         idx = sumpos_or_sum_neg_is_zero
    #         push_sum = torch.sum(push_kernel, dim=(1, 2, 3), keepdims=True)
    #         pull_sum = torch.sum(pull_kernel, dim=(1, 2, 3), keepdims=True)
    #         pull_kernel[idx] = pull_kernel[idx] * push_sum[idx] / pull_sum[idx]
    #     else:
    #         shift = sumpos
    #         s = torch.sum(torch.where(pull_kernel > 0, pull_kernel, 0), dim=(1, 2, 3))
    #         interval = 0.001 * (2 * (s < shift) - 1)
    #
    #         filter_ids = torch.argwhere(torch.logical_not(sumpos_or_sum_neg_is_zero))
    #         for idx in filter_ids:
    #             counter = 1
    #             while counter < 2:
    #                 pull_kernel[idx] = pull_kernel[idx] + interval[idx]
    #                 s0 = torch.sum(torch.where(pull_kernel[idx] > 0, pull_kernel[idx], 0))
    #                 if (s0-shift[idx])*(s[idx]-shift[idx]) < 0:
    #                     break
    #                 else:
    #                     s[idx] = s0
    #                 counter = counter + 1
    #             pos_pull_values = torch.where(pull_kernel[idx] > 0, pull_kernel[idx], 0)
    #             neg_pull_values = torch.where(pull_kernel[idx] < 0, pull_kernel[idx], 0)
    #             sumpullneg = torch.sum(neg_pull_values)
    #             pull_kernel[idx] = (neg_pull_values * sumneg[idx] / sumpullneg) + pos_pull_values
    #
    #     return pull_kernel

    # def normalize_pull_kernel(self, push_kernel, pull_kernel):
    #     pos = push_kernel > 0
    #     neg = push_kernel < 0
    #     pull_kernel_norm = pull_kernel
    #
    #     is_pos_empty = torch.logical_not(torch.any(pos.view(pos.shape[0], -1), dim=1))
    #     is_neg_empty = torch.logical_not(torch.any(neg.view(neg.shape[0], -1), dim=1))
    #     kernel_is_all_pos_or_all_neg = torch.logical_or(is_pos_empty, is_neg_empty)
    #
    #     if torch.any(kernel_is_all_pos_or_all_neg):
    #         idx = kernel_is_all_pos_or_all_neg
    #         push_sum = torch.abs(torch.sum(push_kernel, dim=(1, 2, 3), keepdims=True))
    #         pull_sum = torch.abs(torch.sum(pull_kernel, dim=(1, 2, 3), keepdims=True))
    #         pull_kernel_norm[idx] = pull_kernel[idx] / pull_sum[idx] * push_sum[idx]
    #     else:
    #         idx = torch.logical_not(kernel_is_all_pos_or_all_neg)
    #         sumpos = torch.abs(torch.sum(torch.where(pos, push_kernel, 0), dim=(1, 2, 3), keepdim=True))
    #         sumneg = torch.abs(torch.sum(torch.where(neg, push_kernel, 0), dim=(1, 2, 3), keepdim=True))
    #
    #         pospull = torch.where(pull_kernel > 0, pull_kernel_norm, 0)
    #         negpull = torch.where(pull_kernel < 0, pull_kernel_norm, 0)
    #         pospull_sum = torch.abs(torch.sum(pospull, dim=(1, 2, 3), keepdims=True))
    #         negpull_sum = torch.abs(torch.sum(negpull, dim=(1, 2, 3), keepdims=True))
    #
    #         pull_kernel_norm_pos = pospull / pospull_sum * sumpos
    #         pull_kernel_norm_neg = negpull / negpull_sum * sumneg
    #
    #         pull_kernel_norm[idx] = pull_kernel_norm_pos[idx] + pull_kernel_norm_neg[idx]
    #
    #     return pull_kernel_norm


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


def plot_push_kernels(plot_data, title=None):
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
    # if title:
    #     plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()

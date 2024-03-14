import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F


def push_pull(x, push_kernel, avg=0, alpha=1):
    # push_kernel = self.push_conv.weight
    min_push = torch.amin(push_kernel, dim=(2, 3), keepdim=True)
    max_push = torch.amax(push_kernel, dim=(2, 3), keepdim=True)
    pull_kernel = -push_kernel + (max_push + min_push)
    push_sum = torch.sum(push_kernel, dim=(2, 3), keepdims=True)
    pull_sum = torch.sum(pull_kernel, dim=(2, 3), keepdims=True)
    pull_kernel = pull_kernel / pull_sum * push_sum

    # push_response = self.push_conv(x)
    push_response = F.conv2d(x, push_kernel, bias=None, stride=1, padding=0, dilation=0, groups=0)
    pull_response = F.conv2d(x, pull_kernel, bias=None, stride=1, padding=0, dilation=0, groups=0)

    if avg:
        pull_response = F.avg_pool2d(pull_response, avg, count_include_pad=False)
    push_response = F.relu_(push_response)
    pull_response = F.relu_(pull_response)

    x_out = push_response - pull_response * alpha

    return x_out


def plot_kernels(ckpt):
    model = torch.load(ckpt)
    state_dict = model['state_dict']
    alpha = state_dict['conv1.pull_inhibition_strength'].detach().cpu().numpy()
    bias = state_dict['conv1.bias'].detach().cpu().numpy()
    push_kernel = state_dict['conv1.push_conv.weight'].detach().cpu().numpy()
    inhibition = state_dict['conv1.pull_inhibition_strength'].detach().cpu().numpy()
    mat = {
        'push': state_dict['conv1.weight'].detach().cpu().numpy(),
        # 'alpha': inhibition
    }
    scipy.io.savemat(
        '/scratch/p288722/runtime_data/pushpull-conv/resnet18_imagenet_classification/resnet18_imagenet.mat', mat)

    plt.bar(range(64), inhibition)
    plt.title(f'Learned Inhibition Strength\n'
              f'{model["hyper_parameters"]["dataset_name"]} - '
              f'{model["hyper_parameters"]["model"]} avg{model["hyper_parameters"]["avg_kernel_size"]}')
    plt.show()

    filter_ids = [0, ]
    channel_id = 0
    selected_filters = []

    pass


if __name__ == '__main__':
    plot_kernels(
        ckpt='/scratch/p288722/runtime_data/pushpull-conv/resnet18_imagenet_classification/resnet18/checkpoints/epoch=19-last.ckpt'
    )

    # Multi-hop SSH to connect to the dedicated compute nodes of ULE HPC Cluster
    # ssh -J guru.swaroop@193.146.98.96 -p 10001 guru@computo04.unileon.hpc
    # ssh -J guru.swaroop@193.146.98.96 -p 10001 guru@computo12.unileon.hpc

    # Setting up local port forwarding
    # ssh -o ExitOnForwardFailure=yes -fN -J guru.swaroop@193.146.98.96 -L 10004:localhost:22 -p 10001 guru@computo04.unileon.hpc
    # ssh -o ExitOnForwardFailure=yes -fN -J guru.swaroop@193.146.98.96 -L 10012:localhost:22 -p 10001 guru@computo12.unileon.hpc

    # View all the ports in use
    # lsof -i -P -n | grep LISTEN

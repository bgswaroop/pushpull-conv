from pathlib import Path

import PIL.Image
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.models import resnet50, ResNet50_Weights


def plot_DoG():
    def gaus2d(x=0, y=0, mean=0, std=1):
        sx = sy = std
        mx = my = mean
        return 1. / (2. * np.pi * sx * sy) * np.exp(
            -((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))

    x = np.linspace(-8, 8, num=100)
    y = np.linspace(-8, 8, num=100)
    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D
    z1 = gaus2d(x, y, mean=0, std=2)
    z2 = gaus2d(x, y, mean=0, std=4)
    z = z1 - z2

    fig = plt.figure(dpi=300, figsize=(4, 3))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z1, cmap=cm.jet)
    ax.set_zlim(-0.1, 0.03)
    # plt.tight_layout()
    plt.title(f'Gaussian Conv')
    # plt.savefig('Gauss.png')
    plt.show()

    fig = plt.figure(dpi=300, figsize=(4, 3))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.jet)
    ax.set_zlim(-0.1, 0.03)
    # plt.tight_layout()
    plt.title(f'DoG Conv')
    # plt.savefig('DoG.png')
    plt.show()


def plot_PushPull(alpha=0.5, avg=3):
    def gaus2d(x=0, y=0, mean=0, std=1):
        sx = sy = std
        mx = my = mean
        return 1. / (2. * np.pi * sx * sy) * np.exp(
            -((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))

    x = np.linspace(-8, 8, num=100)
    y = np.linspace(-8, 8, num=100)
    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D
    push_conv = gaus2d(x, y, mean=0, std=2)
    pull_conv = -push_conv + np.max(push_conv) + np.min(push_conv)
    pull_conv = F.conv2d(input=torch.tensor(pull_conv).unsqueeze(0).unsqueeze(0).float(),
                         weight=(torch.ones((avg, avg)) / avg ** 2).unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
    z = push_conv - alpha * pull_conv.detach().numpy()

    fig = plt.figure(dpi=300, figsize=(4, 3))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, np.zeros_like(x), rstride=1, cstride=1, alpha=0.1, color='gray')
    ax.plot_surface(x[2:-2, 2:-2], y[2:-2, 2:-2], z[2:-2, 2:-2], cmap=cm.jet, alpha=0.8)
    ax.set_zlim(-0.1, 0.03)
    # plt.tight_layout()
    plt.title(rf'PuPu-Conv, $\alpha=${alpha}')
    # plt.savefig(f'pp_inh{alpha}.png')
    plt.show()


def push_pull(push_kernel, x, avg=3, alpha=1):
    W = push_kernel
    min_push = torch.amin(W, dim=(1, 2, 3), keepdim=True)
    max_push = torch.amax(W, dim=(1, 2, 3), keepdim=True)
    pull_kernel = -W + (max_push + min_push)

    push_response = F.conv2d(x, push_kernel, stride=2, padding=3)
    pull_response = F.conv2d(x, pull_kernel, stride=2, padding=3)

    if avg:
        pull_response = F.avg_pool2d(pull_response, kernel_size=avg, stride=1, padding=1)

    push_response = F.relu_(push_response)
    pull_response = F.relu_(pull_response)
    x_out = push_response - pull_response * alpha
    return x_out


def old_push_pull(push_kernel, x, alpha=1):
    W = push_kernel
    pull_kernel = -F.upsample(W, size=(15, 15), align_corners=True, mode='bilinear')

    push_response = F.conv2d(x, push_kernel, stride=2, padding=3)
    pull_response = F.conv2d(x, pull_kernel, stride=2, padding=7)

    push_response = F.relu_(push_response)
    pull_response = F.relu_(pull_response)
    x_out = push_response - pull_response * alpha
    return x_out


def analyze():
    model = resnet50(ResNet50_Weights.IMAGENET1K_V1)
    input = np.asarray(PIL.Image.open(
        r'/home/guru/datasets/imagenet/imagenet-c/gaussian_noise/1/n01820546/ILSVRC2012_val_00045836.JPEG'))
    input = input.transpose([2, 0, 1])
    input = np.expand_dims(input, 0) / 255.0
    input = torch.tensor(input).float()

    num_filters = 64
    fig, ax = plt.subplots(num_filters, 7, figsize=(7 + 5, num_filters), dpi=200)
    fig.subplots_adjust(hspace=0, wspace=0)
    for filter_id in range(num_filters):
        push_kernel = torch.unsqueeze(model.conv1.weight[filter_id], dim=0)
        new_pp_out = push_pull(push_kernel, x=input, avg=3, alpha=1)
        old_pp_out = old_push_pull(push_kernel, x=input, alpha=1)
        push_response = F.conv2d(input, push_kernel, stride=2, padding=3)

        im = [None] * 7
        im[0] = ax[filter_id][0].imshow(push_kernel[0, 0].detach().numpy())
        im[1] = ax[filter_id][1].imshow(push_kernel[0, 1].detach().numpy())
        im[2] = ax[filter_id][2].imshow(push_kernel[0, 2].detach().numpy())
        im[3] = ax[filter_id][3].imshow(input[0].detach().numpy().transpose([1, 2, 0]))
        im[4] = ax[filter_id][4].imshow(push_response[0, 0].detach().numpy())
        im[5] = ax[filter_id][5].imshow(new_pp_out[0, 0].detach().numpy())
        im[6] = ax[filter_id][6].imshow(old_pp_out[0, 0].detach().numpy())

        for idx in range(7):
            ax[filter_id][idx].get_xaxis().set_ticks([])
            ax[filter_id][idx].get_yaxis().set_ticks([])

            divider = make_axes_locatable(ax[filter_id][idx])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im[idx], cax=cax)

        ax[filter_id][0].set_ylabel(f'{filter_id}')
        if filter_id == 0:
            ax[filter_id][0].set_title('Push kernel\nchannel 0', fontsize=9)
            ax[filter_id][1].set_title('Push kernel\nchannel 1', fontsize=9)
            ax[filter_id][2].set_title('Push kernel\nchannel 2', fontsize=9)
            ax[filter_id][3].set_title('Input\nimage', fontsize=9)
            ax[filter_id][4].set_title('Push\nresponse', fontsize=9)
            ax[filter_id][5].set_title('New PP\nresponse', fontsize=9)
            ax[filter_id][6].set_title('Old PP\nresponse', fontsize=9)

    # plt.tight_layout()
    plt.show()
    print(' ')


if __name__ == '__main__':

    # test_surf()

    # plot_DoG()
    # plot_PushPull(alpha=0.3)
    # plot_PushPull(alpha=0.6)
    # plot_PushPull(alpha=1)
    # plot_PushPull(alpha=2)
    # analyze()

    data = dict()
    basedirs = \
        [
            Path(r'/home/guru/runtime_data/pushpull-conv/resnet18_cifar10_classification_w_relu'),
            Path(r'/home/guru/runtime_data/pushpull-conv/resnet18_imagenet_classification_w_relu'),
            Path(r'/home/guru/runtime_data/pushpull-conv/resnet50_cifar10_classification_w_relu'),

            # Path(r'/home/guru/runtime_data/pushpull-conv/resnet18_imagenet100_classification_w_relu'),
            # Path(r'/home/guru/runtime_data/pushpull-conv/resnet18_imagenet200_classification_w_relu'),
            # Path(r'/home/guru/runtime_data/pushpull-conv/resnet50_imagenet100_classification_w_relu'),
            # Path(r'/home/guru/runtime_data/pushpull-conv/resnet50_imagenet200_classification_w_relu'),
            # Path(r'/home/guru/runtime_data/pushpull-conv/resnet50_imagenet_classification_w_relu'),
        ]
    for basedir in basedirs:
        file = basedir.joinpath(rf'{basedir.name.split("_")[0]}_avg3_inh_trainable/checkpoints/last.ckpt')
        model = torch.load(file)
        name = f'{"_".join(basedir.name.split("_")[:2])}_avg3_inh_trainable'
        data[name] = model['state_dict']['conv1.push_conv.weight'].detach().cpu().numpy()
        name = f'{"_".join(basedir.name.split("_")[:2])}_avg3_inh_trainable_alpha'
        data[name] = model['state_dict']['conv1.pull_inhibition_strength'].detach().cpu().numpy()

        file = basedir.joinpath(rf'{basedir.name.split("_")[0]}_avg5_inh_trainable/checkpoints/last.ckpt')
        model = torch.load(file)
        name = f'{"_".join(basedir.name.split("_")[:2])}_avg5_inh_trainable'
        data[name] = model['state_dict']['conv1.push_conv.weight'].detach().cpu().numpy()
        name = f'{"_".join(basedir.name.split("_")[:2])}_avg5_inh_trainable_alpha'
        data[name] = model['state_dict']['conv1.pull_inhibition_strength'].detach().cpu().numpy()

    scipy.io.savemat(f'pushpull_classification1.mat', data)

    print('Run finished!')

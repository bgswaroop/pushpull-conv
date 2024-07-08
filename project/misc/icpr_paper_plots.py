import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import StrMethodFormatter
from torchvision.models import resnet50, ResNet50_Weights


def plot_mCE_versus_clean_error():
    filename = '/home/guru/runtime_data/pushpull-conv-ver0/resnet18_imagenet100_classification_w_relu/absolute_CE_top1.csv'
    df = pd.read_csv(filename, index_col=0)
    plt.figure(dpi=300, figsize=(5, 2.5))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
    markers = ["o", "v", "^", "<", ">", "p", "P", "H", '*', "v", "^", "<", ">", "p", "P", "H", '*']
    colors = ['#bf4342',
              # '#61A5C2', '#468FAF', '#2C7DA0', '#2A6F97', '#014F86', '#01497C',
              '#9BAFD9', '#849BCB', '#6D87BC', '#5673AE', '#3E5FA0', '#274B91', '#103783',
              '#003554',
              # '#FFFF00', '#FFE800', '#FFD100', '#FFBA00', '#FFA300', '#FF8C00',
              # https://coolors.co/palette/f9e405-fad507-fac609-fca10a-fd7c0b-fc6b07-fb5a03
              '#F9E405', '#fad507', '#fac609', '#fca10a', '#fd7c0b', '#fc6b07', '#fb5a03',
              '#FF4d00',
              ]

    plt.plot([0.280, 0.200], [0.508, 0.588], '--', alpha=0.3, c='red', linewidth=0.5, label='Trade-off with beta=2.81')
    plt.text(0.245, 0.53, '+ve', c='red', alpha=0.3, size=6)
    plt.text(0.25, 0.545, '-ve', c='red', alpha=0.3, size=6)

    plt.plot([0.235, 0.205], [0.503, 0.587], '--', alpha=0.3, c='blue', linewidth=0.5, label='Trade-off with beta=1')
    plt.text(0.22, 0.51, '+ve', c='blue', alpha=0.3, size=6)
    plt.text(0.235, 0.52, '-ve', c='blue', alpha=0.3, size=6)

    for idx, (ce, mce, l, m, c) in enumerate(zip(df['Error'], df['mCE'], df.index, markers, colors)):
        if idx == 0:
            prefix_to_discard = f'{l}_'
            l = 'baseline'
            x0 = ce
            y0 = mce
        else:
            l = l[len(prefix_to_discard):]
        # tradeoff_ratio1 = (1 - ce) / mce
        # tradeoff_ratio2 = (1 - mce) / ce
        # print(f'{round(tradeoff_ratio1, 3)}, {round(tradeoff_ratio2, 3)} - {l}')
        print(f'{l}, {ce}, {mce}')
        plt.scatter(ce, mce, s=5**2, marker=m, label=l, c=c, alpha=0.6)

    # x + y = x0 + y0
    # 2 * x0 * y0 = x0 * y + y0 *x




    # plt_title = 'ResNet18 trained on ImageNet100'
    # plt.title(plt_title, pad=12, fontsize=11)
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)

    plt.xlabel('Clean error', labelpad=None, fontsize=6)
    plt.ylabel('Absolute mCE', labelpad=None, fontsize=6)
    plt.legend(bbox_to_anchor=(1.05, 1.00), loc="upper left", prop={'size': 6}, frameon=False)
    plt.tight_layout(pad=0.5)
    plt.savefig(f'_plot_figures/figure_{Path(filename).parent.name}.eps', format='eps')
    # plt.savefig(f'_plot_figures/figure_{Path(filename).parent.name}_30epochs.png')
    plt.show()


def plot_shot_noise_corruptions_cifar_10():
    # Accuracy [clean, severity1, severity2, severity1, severity4, severity5]
    resnet18 = [0.9312999844551086, 0.8461999893188477, 0.7534999847412109,
                0.5242999792098999, 0.4392000138759613, 0.31700000166893005]
    resnet18_pp_avg5 = [0.928600013256073, 0.8647000193595886, 0.7890999913215637,
                        0.5820000171661377, 0.5041000247001648, 0.38350000977516174]
    resnet50 = [0.9312999844551086, 0.8341000080108643, 0.7405999898910522,
                0.520799994468689, 0.4268999993801117, 0.303600013256073]
    resnet50_pp_avg5 = [0.9323999881744385, 0.8571000099182129, 0.7796000242233276,
                        0.5805000066757202, 0.49160000681877136, 0.3596000075340271]

    severity = [0, 1, 2, 3, 4, 5]

    # from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
    # imagebox = OffsetImage(img, zoom=0.15)
    # ab = AnnotationBbox(imagebox, (5, 700), frameon=False)
    # ax.add_artist(ab)

    def unpickle(file):
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            X = data[b'data']
        return X

    clean_cifar_10 = unpickle('/home/guru/datasets/cifar/cifar-10-batches-py/test_batch')

    shot_noise = np.load('/home/guru/datasets/cifar/cifar10-c/shot_noise.npy')
    img_index = 110  # should be [0,9999]
    images = [None] * 6
    for idx in range(6):
        if idx == 0:
            r = clean_cifar_10[img_index][0:1024].reshape((32, 32))
            g = clean_cifar_10[img_index][1024:2048].reshape((32, 32))
            b = clean_cifar_10[img_index][2048:3072].reshape((32, 32))
            images[idx] = np.dstack([r, g, b])
        else:
            images[idx] = shot_noise[img_index + (idx - 1) * 10000]

    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5, 3.5))
    ax.plot(severity, resnet18, marker='.', ls='--', color='#2a9d8f', label='ResNet18')
    ax.plot(severity, resnet18_pp_avg5, marker='.', ls='-', color='#2a9d8f', label='ResNet18-PushPull')
    ax.plot(severity, resnet50, marker='.', ls='--', color='#fca311', label='ResNet50')
    plt.plot(severity, resnet50_pp_avg5, marker='.', ls='-', color='#fca311', label='ResNet50-PushPull')

    import cv2



    for idx, xy in [
        (0, (0.3, 0.72)),
        (1, (1.3, 0.6)),
        (2, (2.5, 0.85)),
        (3, (2.8, 0.38)),
        (4, (3.5, 0.65)),
        (5, (4.65, 0.50))
    ]:
        img = cv2.resize(images[idx], dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
        imagebox = OffsetImage(img)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, xy=xy,
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0,
                            frameon=False, )
        ax.add_artist(ab)

    plt.tick_params(axis='x', labelsize=7)
    plt.tick_params(axis='y', labelsize=7)
    plt.title('ResNet models trained on CIFAR-10', pad=12, fontsize=11)

    plt.xlabel('Shot Noise Corruption Severity (CIFAR10-C)', labelpad=8, fontsize=10)
    plt.ylabel('Classification Accuracy', labelpad=8, fontsize=10)
    params = {'legend.fontsize': 6,
              'legend.handlelength': 3}
    plt.rcParams.update(params)
    plt.legend(fontsize=8, loc="upper right", frameon=True)
    plt.tight_layout(pad=0.5)

    # plt.show()
    # plt.savefig(f'_plot_figures/figure_illustration_of_image_corruption.eps', format='eps')
    # plt.savefig(f'_plot_figures/figure_illustration_of_image_corruption.png', format='png')
    plt.show()


def plot_push_pull_kernels():
    model = resnet50(ResNet50_Weights.IMAGENET1K_V1)
    push = model.conv1.weight.detach().cpu().numpy().transpose((0, 2, 3, 1))[:, :, :, 0]
    pull = -push + push.min((1, 2), keepdims=True) + push.max((1, 2), keepdims=True)

    # Plot all Push Kernels
    fig, ax = plt.subplots(8, 8, dpi=300, figsize=(8, 8))
    for row_id in range(8):
        for col_id in range(8):
            ax[row_id][col_id].imshow(push[col_id + 8 * row_id])
            ax[row_id][col_id].get_xaxis().set_ticks([])
            ax[row_id][col_id].get_yaxis().set_ticks([])
    plt.show()

    # Plot 5 push and pull kernels
    filter_ids = [1, 5, 7, 27, 56]
    fig, ax = plt.subplots(2, 5, dpi=300, figsize=(5, 2))
    for col_id in range(5):
        ax[0][col_id].imshow(push[filter_ids[col_id]])
        ax[1][col_id].imshow(pull[filter_ids[col_id]])
        ax[0][col_id].get_xaxis().set_ticks([])
        ax[0][col_id].get_yaxis().set_ticks([])
        ax[1][col_id].get_xaxis().set_ticks([])
        ax[1][col_id].get_yaxis().set_ticks([])
    plt.show()


def plot_fourier_analysis_of_pushpull_filters():
    model = resnet50(ResNet50_Weights.IMAGENET1K_V1)
    push = model.conv1.weight.detach().cpu().numpy().transpose((0, 2, 3, 1))[:, :, :, 0]
    pull = -push + push.min((1, 2), keepdims=True) + push.max((1, 2), keepdims=True)

    data = {
        'push': push,
        'pull': pull
    }
    scipy.io.savemat(f'/home/guru/git_code/pushpull-conv/project/misc/pushpull_resnet50.mat', data)
    # The analysis was continued in MATLAB

import torch
def plot_learnable_inhibition_strengths():
    r50_avg3 = torch.load(r'/home/guru/runtime_data/pushpull-conv/resnet50_imagenet_classification_w_relu/'
                          r'resnet50_avg3_inh_trainable/checkpoints/last.ckpt')['state_dict']['conv1.pull_inhibition_strength']
    r50_avg5 = torch.load(r'/home/guru/runtime_data/pushpull-conv/resnet50_imagenet_classification_w_relu/'
                          r'resnet50_avg5_inh_trainable/checkpoints/last.ckpt')['state_dict']['conv1.pull_inhibition_strength']
    r18_avg3 = torch.load(r'/home/guru/runtime_data/pushpull-conv/resnet18_imagenet_classification_w_relu/'
                          r'resnet18_avg3_inh_trainable/checkpoints/last.ckpt')['state_dict'][
        'conv1.pull_inhibition_strength']
    r18_avg5 = torch.load(r'/home/guru/runtime_data/pushpull-conv/resnet18_imagenet_classification_w_relu/'
                          r'resnet18_avg5_inh_trainable/checkpoints/last.ckpt')['state_dict'][
        'conv1.pull_inhibition_strength']
    fig, ax = plt.subplots(2, 2, dpi=300, figsize=(5, 3), sharex=False, sharey=True)
    ax[0][0].hist(r18_avg3.detach().cpu().numpy(), bins=40, log=True)
    ax[0][1].hist(r18_avg5.detach().cpu().numpy(), bins=40, log=True)
    ax[1][0].hist(r50_avg3.detach().cpu().numpy(), bins=40, log=True)
    ax[1][1].hist(r50_avg5.detach().cpu().numpy(), bins=40, log=True)
    ax[0][0].tick_params(axis='x', labelsize=6)
    ax[0][0].tick_params(axis='y', labelsize=6)
    ax[0][1].tick_params(axis='x', labelsize=6)
    ax[0][1].tick_params(axis='y', labelsize=6)
    ax[1][0].tick_params(axis='x', labelsize=6)
    ax[1][0].tick_params(axis='y', labelsize=6)
    ax[1][1].tick_params(axis='x', labelsize=6)
    ax[1][1].tick_params(axis='y', labelsize=6)

    ax[0][0].set_title('ResNet18 avg3', size=9)
    ax[0][1].set_title('ResNet18 avg5', size=9)
    ax[1][0].set_title('ResNet50 avg3', size=9)
    ax[1][1].set_title('ResNet50 avg5', size=9)
    plt.tight_layout()
    plt.savefig(f'_plot_figures/figure_distribution_of_alpha.eps', format='eps')
    plt.show()

    fig, ax = plt.subplots(2, 1, dpi=300, figsize=(5, 3), sharex=False, sharey=True)
    ax[0][0].hist(r18_avg3.detach().cpu().numpy(), bins=40, log=True, color='red')
    ax[0][0].hist(r18_avg5.detach().cpu().numpy(), bins=40, log=True, color='green')
    ax[1][0].hist(r50_avg3.detach().cpu().numpy(), bins=40, log=True, color='red')
    ax[1][0].hist(r50_avg5.detach().cpu().numpy(), bins=40, log=True, color='green')
    ax[0][0].tick_params(axis='x', labelsize=6)
    ax[0][0].tick_params(axis='y', labelsize=6)
    # ax[0][1].tick_params(axis='x', labelsize=6)
    # ax[0][1].tick_params(axis='y', labelsize=6)
    ax[1][0].tick_params(axis='x', labelsize=6)
    ax[1][0].tick_params(axis='y', labelsize=6)
    # ax[1][1].tick_params(axis='x', labelsize=6)
    # ax[1][1].tick_params(axis='y', labelsize=6)

    # ax[0][0].set_title('ResNet18 avg3', size=9)
    # ax[0][1].set_title('ResNet18 avg5', size=9)
    # ax[1][0].set_title('ResNet50 avg3', size=9)
    # ax[1][1].set_title('ResNet50 avg5', size=9)
    plt.tight_layout()
    plt.savefig(f'_plot_figures/figure_distribution_of_alpha.eps', format='eps')
    plt.show()

    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(2, 1, dpi=300, figsize=(5, 3), sharex=True, sharey=True)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # ax[0].hist(r18_avg3.detach().cpu().numpy(), bins=40, log=True, color='red', alpha=0.4, label='ResNet18 - AvgPool3')
    # ax[0].hist(r18_avg5.detach().cpu().numpy(), log=True, color='green', alpha=0.4, label='ResNet18 - AvgPool5')
    # ax[1].hist(r50_avg3.detach().cpu().numpy(), bins=40, log=True, color='red', alpha=0.4, label='ResNet50 - AvgPool3')
    # ax[1].hist(r50_avg5.detach().cpu().numpy(), log=True, color='green', alpha=0.4, label='ResNet50 - AvgPool5')
    #
    # ax[0].legend()
    # ax[1].legend()
    #
    # # ax[0].tick_params(axis='x', labelsize=6)
    # ax[0].tick_params(axis='y', labelsize=8)
    # # ax[0][1].tick_params(axis='x', labelsize=6)
    # # ax[0][1].tick_params(axis='y', labelsize=6)
    # ax[1].tick_params(axis='x', labelsize=8)
    # ax[1].tick_params(axis='y', labelsize=8)
    # # ax[1][1].tick_params(axis='x', labelsize=6)
    # # ax[1][1].tick_params(axis='y', labelsize=6)
    # # ax[0][0].set_title('ResNet18 avg3', size=9)
    # # ax[0][1].set_title('ResNet18 avg5', size=9)
    # # ax[1][0].set_title('ResNet50 avg3', size=9)
    # # ax[1][1].set_title('ResNet50 avg5', size=9)
    #
    # plt.tight_layout()
    # plt.savefig(f'_plot_figures/figure_distribution_of_alpha.png', format='png')
    # plt.show()

    print(' ')

if __name__ == '__main__':
    plot_mCE_versus_clean_error()
    # plot_shot_noise_corruptions_cifar_10()
    # plot_push_pull_kernels()
    # plot_fourier_analysis_of_pushpull_filters()
    # plot_learnable_inhibition_strengths()
    print('Run finished!')

from pathlib import Path

import torch
from matplotlib import pyplot as plt


def plot_trainable_inhibitions():
    # ckpt = Path('/home/guru/runtime_data/pushpull-conv/resnet50_imagenet_classification_w_relu/resnet50_avg5_inh_trainable/checkpoints/last.ckpt')
    ckpt = Path(
        r'/home/guru/runtime_data/pushpull-conv-new/resnet18_imagenet100_classification/resnet18_wo_relu_avg3_inh_trainable/checkpoints/last.ckpt')
    model = torch.load(ckpt)
    x = model['state_dict']['conv1.pull_inhibition_strength'].cpu()
    plt.figure(dpi=150)
    plt.hist(x, bins=20)
    plt.axvline(x=0, ls='--', color='black')
    plt.xlabel('Inhibition values')
    plt.ylabel('Count')
    plt.title(f'{ckpt.parents._parts[5]}\n{ckpt.parents._parts[6]}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_trainable_inhibitions()

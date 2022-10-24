from pathlib import Path

from matplotlib import pyplot as plt
import torch

def plot_trainable_inhibitions():
    # ckpt = Path('/home/guru/runtime_data/pushpull-conv/resnet50_imagenet_classification_w_relu/resnet50_avg5_inh_trainable/checkpoints/last.ckpt')
    ckpt = Path(r'/home/guru/runtime_data/pushpull-conv/resnet50_cifar10_classification_w_relu/resnet50_avg5_inh_trainable_bs64/checkpoints/last.ckpt')
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


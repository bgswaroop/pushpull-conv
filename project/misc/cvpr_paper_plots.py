from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


def plot_mCE_versus_clean_error():
    filename = '/home/guru/runtime_data/pushpull-conv/resnet18_imagenet100_classification/absolute_CE_top1.csv'
    df = pd.read_csv(filename, index_col=0)
    plt.figure(dpi=300, figsize=(5, 3))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
    markers = ["o", "v", "^", "<", ">", "p", "*", "v", "^", "<", ">", "p", "*"]
    colors = ['#bf4342',
              '#61A5C2', '#468FAF', '#2C7DA0', '#2A6F97', '#014F86', '#01497C',
              '#FFFF00', '#FFE800', '#FFD100', '#FFBA00', '#FFA300', '#FF8C00',
              ]
    for idx, (ce, mce, l, m, c) in enumerate(zip(df['Error'], df['mCE'], df.index, markers, colors)):
        if idx == 0:
            prefix_to_discard = f'{l}_pp7x7_'
            l = 'baseline'
        else:
            l = l[len(prefix_to_discard):]
        plt.scatter(ce, mce, marker=m, label=l, c=c)
    plt_title = 'ResNet18 trained on ImageNet100'
    plt.title(plt_title, pad=12, fontsize=11)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)

    plt.xlabel('Clean error', labelpad=8, fontsize=10)
    plt.ylabel('Absolute mCE', labelpad=8, fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1.06), loc="upper left", prop={'size': 8}, frameon=False)
    plt.tight_layout(pad=0.5)
    # plt.savefig(f'_plot_figures/figure_{Path(filename).parent.name}.eps', format='eps')
    plt.savefig(f'_plot_figures/figure_{Path(filename).parent.name}_30epochs.png')
    plt.show()


if __name__ == '__main__':
    plot_mCE_versus_clean_error()
    print('Run finished!')

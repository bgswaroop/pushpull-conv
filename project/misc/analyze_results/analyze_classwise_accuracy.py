import pandas as pd
from matplotlib import pyplot as plt


def main():
    # paths in A40 server
    img100_df = pd.read_csv(index_col=0, usecols=[0] + list(range(2, 78)),
                            filepath_or_buffer='/data2/p288722/runtime_data/pushpull-conv/from_peregrine/resnet50_imagenet100_classification/resnet50/results/classwise_scores.csv')
    img200_df = pd.read_csv(index_col=0, usecols=[0] + list(range(2, 78)),
                            filepath_or_buffer='/data2/p288722/runtime_data/pushpull-conv/from_peregrine/resnet50_imagenet200_classification/resnet50/results/classwise_scores.csv')
    img1k_df = pd.read_csv(index_col=0, usecols=[0] + list(range(2, 78)),
                           filepath_or_buffer='/data2/p288722/runtime_data/pushpull-conv/from_computo12/resnet50_imagenet_classification_60epochs/resnet50/results/classwise_scores.csv')

    img1k_100_df = img1k_df.loc[img1k_df.index & img100_df.index]
    img1k_200_df = img1k_df.loc[img1k_df.index & img200_df.index]
    print(' ')
    print(f'ImageNet1k error when evaluated on ImageNet100 classes (ImageNet test): {1 - img1k_100_df["Clean"].mean()}')
    print(f'ImageNet1k error when evaluated on ImageNet200 classes (ImageNet test): {1 - img1k_200_df["Clean"].mean()}')

    print(
        f'ImageNet1k error when evaluated on ImageNet100 classes (ImageNet-C): {1 - img1k_100_df.iloc[:, 1:-1].mean(axis=1).mean()}')
    print(
        f'ImageNet1k error when evaluated on ImageNet200 classes (ImageNet-C): {1 - img1k_200_df.iloc[:, 1:-1].mean(axis=1).mean()}')

    x = (img1k_100_df['Clean'] - img100_df['Clean']) * 100
    num_low_performance = x[x < 0].count()
    num_high_performance = x[x > 0].count()
    plt.figure(figsize=(8, 4))
    h = plt.hist(x, bins=20)
    plt.axvline(x=0, ls='--', c='black')

    plt.text(min(h[1]) + 5, max(h[0]) - 2, f'{num_low_performance} classes\n declined in accuracy',
             horizontalalignment='center')
    plt.text(max(h[1]) - 5, max(h[0]) - 2, f'{num_high_performance} classes\n improved in accuracy',
             horizontalalignment='center')
    plt.xlabel('Change in accuracy (in percent) from ImageNet100 to ImageNet1k\n'
               'for the 100 classes in ImageNet100')
    plt.ylabel('Number of classes')
    plt.title('Performance of ResNet50 (baseline)\non Clean test images')
    plt.tight_layout()
    plt.show()


    x = (img1k_200_df['Clean'] - img200_df['Clean']) * 100
    num_low_performance = x[x < 0].count()
    num_high_performance = x[x > 0].count()
    plt.figure(figsize=(8, 4))
    h = plt.hist(x, bins=20)
    plt.axvline(x=0, ls='--', c='black')

    plt.text(min(h[1]) + 5, max(h[0]) - 4, f'{num_low_performance} classes\n declined in accuracy', horizontalalignment='center')
    plt.text(max(h[1]) - 5, max(h[0]) - 4, f'{num_high_performance} classes\n improved in accuracy', horizontalalignment='center')
    plt.xlabel('Change in accuracy (in percent) from ImageNet200 to ImageNet1k\n'
               'for the 200 classes in ImageNet200')
    plt.ylabel('Number of classes')
    plt.title('Performance of ResNet50 (baseline)\non Clean test images')
    plt.tight_layout()
    plt.show()

    x = (img1k_100_df.iloc[:, 1:-1].mean(axis=1) - img100_df.iloc[:, 1:-1].mean(axis=1)) * 100
    num_low_performance = x[x < 0].count()
    num_high_performance = x[x > 0].count()
    plt.figure(figsize=(8, 4))
    h = plt.hist(x, bins=20)
    plt.axvline(x=0, ls='--', c='black')

    plt.text(min(h[1]) + 3, max(h[0]) - 1, f'{num_low_performance} classes\n declined in accuracy',
             horizontalalignment='center')
    plt.text(max(h[1]) - 3, max(h[0]) - 1, f'{num_high_performance} classes\n improved in accuracy',
             horizontalalignment='center')
    plt.xlabel('Change in accuracy (in percent) from ImageNet100 to ImageNet1k\n'
               'for the 100 classes in ImageNet100')
    plt.ylabel('Number of classes')
    plt.title('Performance of ResNet50 (baseline)\non corrupted test images (ImageNet-C)')
    plt.tight_layout()
    plt.show()

    x = (img1k_200_df.iloc[:, 1:-1].mean(axis=1) - img200_df.iloc[:, 1:-1].mean(axis=1)) * 100
    num_low_performance = x[x < 0].count()
    num_high_performance = x[x > 0].count()
    plt.figure(figsize=(8, 4))
    h = plt.hist(x, bins=20)
    plt.axvline(x=0, ls='--', c='black')

    plt.text(min(h[1]) + 5, max(h[0]) - 3, f'{num_low_performance} classes\n declined in accuracy',
             horizontalalignment='center')
    plt.text(max(h[1]) - 5, max(h[0]) - 3, f'{num_high_performance} classes\n improved in accuracy',
             horizontalalignment='center')
    plt.xlabel('Change in accuracy (in percent) from ImageNet200 to ImageNet1k\n'
               'for the 200 classes in ImageNet200')
    plt.ylabel('Number of classes')
    plt.title('Performance of ResNet50 (baseline)\non corrupted test images (ImageNet-C)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

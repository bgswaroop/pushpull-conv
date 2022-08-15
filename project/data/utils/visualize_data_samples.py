import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from project.data import get_dataset
from project.train_flow import parse_args


# Credits - https://gist.github.com/6aravind/9595cad7d532f3e3fcd227053abe0ffd#file-visualization_of_image_classification-py
def visualize_classification(loader, labelMap=None, num_classes=10, num_items=10, pad=4):
    for class_id in range(num_classes):
        data_iter = iter(loader)
        # classwise_data = {x: [] for x in range(num_classes)}
        # num_completed = 0
        # for img, label in data_iter:
        #     label = int(label)
        #     if len(classwise_data[label]) < num_items:
        #         classwise_data[label].append(img)
        #         if len(classwise_data[label]) == num_items:
        #             num_completed += 1
        #     if num_completed == 10:
        #         break
        #
        # imgTensor = torch.concat([classwise_data[y][x] for x in range(num_items) for y in range(num_classes)])
        # labels = torch.tensor([y for x in range(num_items) for y in range(num_classes)])

        # Plot images from the same class
        data = []
        for img, label in data_iter:
            label = int(label)
            if label == class_id and len(data) < num_classes * num_items:
                data.append(img)
            if len(data) == num_classes * num_items:
                break
        imgTensor = torch.concat(data)
        labels = torch.tensor([class_id for _ in range(len(data))])

        # Generate image grid
        grid = make_grid(imgTensor, padding=pad, nrow=num_classes)

        # Permute the axis as numpy expects image of shape (H x W x C)
        grid = grid.permute(1, 2, 0)

        # Get Labels
        labels = [labelMap[lbl.item()] for lbl in labels[:num_items]]

        # Set up plot config
        plt.figure(figsize=(num_classes, num_items), dpi=300)
        plt.axis('off')
        # Plot Image Grid
        plt.imshow(grid)
        # # Plot the image titles
        # fact = 1 + (num_classes) / 100
        # rng = np.linspace(1 / (fact * num_classes), 1 - 1 / (fact * num_classes), num=num_items)
        # for idx, val in enumerate(rng):
        #     plt.figtext(val, 0.85, labels[idx], fontsize=8)
        # Show the plot
        plt.title(f'Train Class {class_id} : {labelMap[class_id]}')
        plt.savefig(f'Class_{class_id}_{labelMap[class_id]}_train.png')
        # plt.show()


def run_flow():
    args = parse_args()

    # load data
    dataset = get_dataset(args.dataset_name, args.dataset_dir, img_size=args.img_size)
    train_loader = dataset.get_train_dataloader(args.batch_size, args.num_workers)
    val_loader = dataset.get_validation_dataloader(args.batch_size, args.num_workers)
    args.num_classes = dataset.get_num_classes()

    # visualize the data
    visualize_classification(train_loader, train_loader.dataset.labels_num_to_txt, args.num_classes, num_items=10)
    # visualize_classification(val_loader, val_loader.dataset.labels_num_to_txt, args.num_classes, num_items=5)


if __name__ == '__main__':
    run_flow()

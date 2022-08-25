from .data_modules import CIFAR10, CIFAR10C, ImageNet, ImageNetC

__all__ = (
    'CIFAR10', 'CIFAR10C',
    'ImageNet', 'ImageNetC',
    'get_dataset'
)


def get_dataset(dataset_name, dataset_dir, img_size=224, num_severities=5):
    if dataset_name == 'cifar10':
        dataset = CIFAR10(dataset_dir, img_size, download=True)
    elif dataset_name == 'CIFAR-10-C-224x224':
        dataset = CIFAR10C(dataset_dir, num_severities)
    elif dataset_name == 'imagenet':
        dataset = ImageNet(dataset_dir, img_size)
    elif dataset_name == 'imagenet100':
        dataset = ImageNet(dataset_dir, img_size, use_subset=100)
    elif dataset_name == 'imagenet200':
        dataset = ImageNet(dataset_dir, img_size, use_subset=200)
    elif dataset_name == 'imagenet-c':
        dataset = ImageNetC(dataset_dir)
    elif dataset_name == 'imagenet100-c':
        dataset = ImageNetC(dataset_dir, use_subset=100)
    elif dataset_name == 'imagenet200-c':
        dataset = ImageNetC(dataset_dir, use_subset=200)
    else:
        raise ValueError('Invalid dataset_name')
    return dataset

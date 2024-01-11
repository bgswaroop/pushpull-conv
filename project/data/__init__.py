import torchvision

from .data_modules import CIFAR10, CIFAR10C, ImageNet, ImageNetC

__all__ = (
    'CIFAR10', 'CIFAR10C',
    'ImageNet', 'ImageNetC',
    'get_dataset'
)


def get_dataset(dataset_name, dataset_dir, augment=None, img_size=224, num_severities=5):
    if 'pc' in dataset_name[-2:]:
        train_data_percent = float(dataset_name[:-2].split('_')[-1])
        assert 1.0 <= train_data_percent <= 99.0, 'data percent must be within [1, 99]'
        train_set_fraction = train_data_percent / 100
        dataset_name = dataset_name.split('_')[0]
    else:
        train_set_fraction = 1.0

    if dataset_name == 'cifar10':
        dataset = CIFAR10(dataset_dir, augment=augment, download=True)
    elif dataset_name == 'cifar10-c':
        dataset = CIFAR10C(dataset_dir, num_severities)
    elif dataset_name == 'imagenet':
        dataset = ImageNet(dataset_dir, img_size, train_set_fraction, augment, num_classes=1000)
    elif dataset_name == 'imagenet100':
        dataset = ImageNet(dataset_dir, img_size, train_set_fraction, augment, num_classes=100)
    elif dataset_name == 'imagenet200':
        dataset = ImageNet(dataset_dir, img_size, train_set_fraction, augment, num_classes=200)
    elif dataset_name == 'imagenet-c':
        dataset = ImageNetC(dataset_dir)
    elif dataset_name == 'imagenet100-c':
        dataset = ImageNetC(dataset_dir, use_subset=100)
    elif dataset_name == 'imagenet200-c':
        dataset = ImageNetC(dataset_dir, use_subset=200)
    else:
        raise ValueError('Invalid dataset_name')
    return dataset


def get_augmentation(augmentation, dataset_name):
    if augmentation == 'none':
        return None
    elif augmentation == 'AugMix':
        return torchvision.transforms.AugMix()
    elif augmentation == 'AutoAug':
        if dataset_name == 'cifar10':
            policy = torchvision.transforms.AutoAugmentPolicy.CIFAR10
        elif 'imagenet' in dataset_name:
            policy = torchvision.transforms.AutoAugmentPolicy.IMAGENET
        else:
            raise ValueError('Invalid dataset_name for the type of augmentation')
        return torchvision.transforms.AutoAugment(policy)
    elif augmentation == 'RandAug':
        return torchvision.transforms.RandAugment()

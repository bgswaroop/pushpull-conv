import torchvision

from .data_modules import CIFAR10, CIFAR10C, ImageNet, ImageNetC
from .prime import get_prime_augmentation_for_imagenet

__all__ = (
    'CIFAR10', 'CIFAR10C',
    'ImageNet', 'ImageNetC',
    'get_dataset', 'get_augmentation',
    'get_prime_augmentation_for_imagenet'
)




def get_dataset(dataset_name, dataset_dir, augment=None, img_size=224, grayscale=False, num_severities=5, model=None):
    if 'pc' in dataset_name[-2:]:
        train_data_percent = float(dataset_name[:-2].split('_')[-1])
        assert 1.0 <= train_data_percent <= 99.0, 'data percent must be within [1, 99]'
        train_set_fraction = train_data_percent / 100
        dataset_name = dataset_name.split('_')[0]
    else:
        train_set_fraction = 1.0

    resize_dims_for_cifar = 224 if model == 'AlexNet' else None

    if dataset_name == 'cifar10':
        dataset = CIFAR10(dataset_dir, grayscale, augment=augment, download=True, resize=resize_dims_for_cifar)
    elif dataset_name == 'cifar10-c':
        dataset = CIFAR10C(dataset_dir, num_severities, grayscale, resize=resize_dims_for_cifar)
    elif dataset_name == 'imagenet':
        dataset = ImageNet(dataset_dir, img_size, train_set_fraction, augment, grayscale, num_classes=1000)
    elif dataset_name == 'imagenet100':
        dataset = ImageNet(dataset_dir, img_size, train_set_fraction, augment, grayscale, num_classes=100)
    elif dataset_name == 'imagenet200':
        dataset = ImageNet(dataset_dir, img_size, train_set_fraction, augment, grayscale, num_classes=200)
    elif dataset_name == 'imagenet-c':
        dataset = ImageNetC(dataset_dir, grayscale)
    elif dataset_name == 'imagenet100-c':
        dataset = ImageNetC(dataset_dir, grayscale, use_subset=100)
    elif dataset_name == 'imagenet200-c':
        dataset = ImageNetC(dataset_dir, grayscale, use_subset=200)
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
    elif augmentation == 'TrivialAugment':
        return torchvision.transforms.TrivialAugmentWide()
    elif augmentation == 'prime':
        if 'imagenet' in dataset_name:
            return get_prime_augmentation_for_imagenet()
        else:
            raise ValueError('Invalid dataset_name for the type of augmentation')
    else:
        raise ValueError('Invalid augmentation')

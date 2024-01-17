import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional, Callable, Tuple, Any

import PIL.Image
import numpy as np
import torch
import torchvision.datasets
from PIL import PngImagePlugin
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, InterpolationMode

# import os
# import lmdb
# import pyarrow as pa

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)


class _CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False, ):
        super().__init__(root, train, transform, target_transform, download)
        self.soft_targets = None
        self.num_classes = 10
        self.labels_num_to_txt = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'bird',
            4: 'cat',
            5: 'deer',
            6: 'dog',
            7: 'frog',
            8: 'horse',
            9: 'ship',
            10: 'truck',
        }

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        soft_target = self.soft_targets[index] if self.soft_targets is not None else -1

        return img, target, soft_target

    def update_soft_targets(self, soft_targets):
        assert len(soft_targets) == len(self.targets)
        self.soft_targets = soft_targets

    @staticmethod
    def get_num_classes():
        return 10


class CIFAR10:
    def __init__(
            self,
            root: str,
            augment=None,
            download: bool = True,
    ) -> None:
        self.root = root
        self.download = download

        _normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        if augment:
            self._transform_train = transforms.Compose([
                augment,
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                _normalize
            ])
        else:
            self._transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                _normalize
            ])
        self._transform_test = transforms.Compose([
            transforms.ToTensor(),
            _normalize
        ])

    def get_train_dataloader(self, batch_size, num_workers, shuffle=True):
        dataset = _CIFAR10(self.root, True, self._transform_train, None, self.download)
        # data_split_train, data_split_val = random_split(dataset, self._split, torch.Generator().manual_seed(99))
        train_loader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers, persistent_workers=True)
        return train_loader

    def get_validation_dataloader(self, batch_size, num_workers, shuffle=False):
        return self.get_test_dataloader(batch_size, num_workers, shuffle)

    def get_test_dataloader(self, batch_size, num_workers, shuffle=False):
        dataset = _CIFAR10(self.root, False, self._transform_test, None, self.download)
        test_dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers,
                                     persistent_workers=True)
        return test_dataloader

    @staticmethod
    def get_num_classes():
        return 10


class CIFAR10C(Dataset):
    def __init__(self, root_dir, num_splits):
        super(CIFAR10C, self).__init__()

        self.num_splits = num_splits

        # full dataset
        corruptions = sorted(Path(root_dir).glob('*'))
        self._corruption_dataset_files = {x.stem: x for x in corruptions if x.stem != 'labels'}
        self._labels = np.split(np.load(Path(root_dir).joinpath('labels.npy')), num_splits)
        self._labels = torch.tensor(np.array(self._labels)).long()

        # corruption types
        self._all_corruption_types = set([x.stem for x in corruptions if x.stem != 'labels'])
        self.val_corruption_types = {'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'}
        self.test_corruption_types = self._all_corruption_types.difference(self.val_corruption_types)

        # the corruption type and severity to consider from the full dataset
        self._corruption_type = next(iter(self._all_corruption_types))
        self._severity_level = 0

        # data transform
        _normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.transform = transforms.Compose([transforms.ToTensor(), _normalize])

    @property
    def corruption_type(self):
        return self._corruption_type

    @corruption_type.setter
    def corruption_type(self, value):
        assert value in self._all_corruption_types, 'Invalid corruption type'
        self._corruption_type = value

    @property
    def severity_level(self):
        return self._severity_level

    @severity_level.setter
    def severity_level(self, value):
        self._severity_level = value - 1
        data = np.load(self._corruption_dataset_files[self._corruption_type])
        self._corruptions = np.split(data, self.num_splits)[value - 1]

    def __getitem__(self, index):
        img = self.transform(self._corruptions[index])
        label = self._labels[self._severity_level][index]
        return img, label, -1

    def __len__(self):
        return len(self._labels[self._severity_level])

    def get_test_dataloader(self, batch_size, num_workers):
        test_dataloader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
        return test_dataloader

    @staticmethod
    def get_num_classes():
        return 10


class _ImageNetBase(Dataset):
    def __init__(
            self,
            root: str,
            split: str,
            img_size: int,
            num_classes: int,
            data_fraction: float = 1.0,
            augment=None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.num_classes = num_classes
        self.data_fraction = data_fraction
        self.soft_targets = None

        _normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if augment:
            self._transform_train = transforms.Compose([
                augment,
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                _normalize
            ])
        else:
            self._transform_train = transforms.Compose([
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                _normalize
            ])
        self._transform_test = transforms.Compose([
            transforms.Resize(img_size, InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            _normalize
        ])
        self.data = []
        self.labels = []

        self._setup()
        # self._lmdb_setup()

    # def _lmdb_setup(self):
    #
    #     if self.split == 'train':
    #         self.transform = self._transform_train
    #     elif self.split == 'val':
    #         self.transform = self._transform_test
    #     self.root = str(self.root.joinpath(self.split))
    #
    #     self.env = lmdb.open(self.root, subdir=os.path.isdir(self.root),
    #                          readonly=True, lock=False,
    #                          readahead=False, meminit=False)
    #     with self.env.begin(write=False) as txn:
    #         self.length = pa.deserialize(txn.get(b'__len__'))
    #         self.keys = pa.deserialize(txn.get(b'__keys__'))

    def _setup(self):
        labels_text = sorted([x.stem for x in self.root.glob('train/*')])
        self.labels_txt_to_num = {x: idx for idx, x in enumerate(labels_text)}
        self.labels_num_to_txt = {idx: x for idx, x in enumerate(labels_text)}

        if self.split == 'train':
            self.data = sorted(self.root.glob('train/*/*'))
            self.labels = [self.labels_txt_to_num[x.stem.split('_')[0]] for x in self.data]
            self.transform = self._transform_train

        elif self.split == 'val':
            self.data = sorted(self.root.glob('val/*/*'))
            self.labels = [self.labels_txt_to_num[x.parent.stem] for x in self.data]
            self.transform = self._transform_test

        else:
            raise ValueError('Invalid split')

        if self.num_classes < 1000:
            self._filter_to_subset_classes()

        if self.data_fraction != 1.0:
            self._select_stratified_data_fraction()

        # Relabel the samples (needed when num_classes < 1000)
        relabel_map = {x: idx for idx, x in enumerate(sorted(np.unique(self.labels)))}
        self.labels = [relabel_map[x] for x in self.labels]
        self.labels_txt_to_num = {self.labels_num_to_txt[old_id]: new_id for old_id, new_id in relabel_map.items()}
        self.labels_num_to_txt = {num: text for text, num in self.labels_txt_to_num.items()}

        self.length = len(self.labels)

    def _filter_to_subset_classes(self, ):
        assert self.num_classes in {100, 200}, 'Invalid number of subset classes!'
        with open(Path(__file__).parent.joinpath(f'imagenet_c/imagenet_{self.num_classes}_classes.json')) as f:
            subset_class_names = sorted(json.load(f))
            subset_class_numbers = set([self.labels_txt_to_num[x] for x in subset_class_names])
        data, labels = [], []
        for d, l in zip(self.data, self.labels):
            if l in subset_class_numbers:
                data.append(d)
                labels.append(l)
        self.data = data
        self.labels = labels

    def _select_stratified_data_fraction(self):
        grouped_data = defaultdict(list)
        for img_path, label in zip(self.data, self.labels):
            grouped_data[label].append(img_path)
        for label, images in grouped_data.items():
            grouped_data[label] = sorted(images)[:int(len(images) * self.data_fraction)]
        data = []
        labels = []
        for label, images in grouped_data.items():
            labels.extend([label] * len(images))
            data.extend(images)
        self.data = data
        self.labels = labels

    def update_soft_targets(self, soft_targets):
        assert len(soft_targets) == len(self.labels)
        self.soft_targets = soft_targets

    def __getitem__(self, index):
        img = PIL.Image.open(self.data[index]).convert('RGB')
        img = self.transform(img)
        label = self.labels[index]
        soft_target = self.soft_targets[index] if self.soft_targets is not None else -1
        return img, label, soft_target

        # env = self.env
        # with env.begin(write=False) as txn:
        #     byteflow = txn.get(self.keys[item])
        # img_buf, target = pa.deserialize(byteflow)
        #
        # # load image
        # img = PIL.Image.open(io.BytesIO(img_buf)).convert('RGB')
        # img = self.transform(img)
        #
        # return img, target

    def __len__(self):
        # return len(self.labels)
        return self.length

    def get_num_classes(self):
        # return np.unique(self.labels).size
        return self.num_classes


class ImageNet:
    def __init__(self, root: str, img_size: int, train_set_fraction=1.0, augment=None, num_classes=1000):
        self.root = root
        self.img_size = img_size
        self.num_classes = num_classes
        self.train_set_fraction = train_set_fraction
        self.augment = augment

    def get_train_dataloader(self, batch_size, num_workers, shuffle=True):
        self.dataset = _ImageNetBase(self.root, 'train', self.img_size, self.num_classes, self.train_set_fraction,
                                     self.augment)
        train_loader = DataLoader(self.dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                  prefetch_factor=16, pin_memory=True, persistent_workers=True)
        return train_loader

    def get_validation_dataloader(self, batch_size=None, num_workers=None, shuffle=False):
        self.dataset = _ImageNetBase(self.root, 'val', self.img_size, self.num_classes)
        val_loader = DataLoader(self.dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                prefetch_factor=16, pin_memory=True, persistent_workers=True)
        return val_loader

    def get_test_dataloader(self, batch_size, num_workers, shuffle=False):
        self.dataset = _ImageNetBase(self.root, 'val', self.img_size, self.num_classes)
        test_loader = DataLoader(self.dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 prefetch_factor=16, pin_memory=True, persistent_workers=True)
        return test_loader

    def get_num_classes(self):
        return self.dataset.get_num_classes()


class ImageNetC:
    def __init__(self, root_dir, use_subset=None):
        super(ImageNetC, self).__init__()

        # full dataset
        self.root = Path(root_dir)
        self.use_subset = use_subset

        # corruption types
        self._all_corruption_types = set([x.stem for x in sorted(self.root.glob('*')) if x.stem != 'labels'])
        self.val_corruption_types = {'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'}
        self.test_corruption_types = self._all_corruption_types.difference(self.val_corruption_types)

        # the corruption type and severity to consider from the full dataset
        self._corruption_type = next(iter(self._all_corruption_types))
        self._severity_level = '1'

        # data transform
        _normalize = transforms.Normalize((0.485, 0.456, 0.406), (00.229, 0.224, 0.225))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            _normalize
        ])

        self.images = []
        self.labels = []

    @property
    def corruption_type(self):
        return self._corruption_type

    @corruption_type.setter
    def corruption_type(self, value):
        assert value in self._all_corruption_types, 'Invalid corruption type'
        self._corruption_type = value

    @property
    def severity_level(self):
        return self._severity_level

    @severity_level.setter
    def severity_level(self, value):
        self._severity_level = str(int(value))
        dataset = self.root.joinpath(self._corruption_type).joinpath(self._severity_level)
        self.images = list(dataset.glob('*/*'))
        labels_text = sorted([x.stem for x in dataset.glob('*')])
        self.labels_txt_to_num = {x: idx for idx, x in enumerate(labels_text)}
        self.labels_num_to_txt = {idx: x for idx, x in enumerate(labels_text)}
        self.labels = [self.labels_txt_to_num[x.parent.name] for x in self.images]

        if self.use_subset:
            self._extract_subset(num_classes=self.use_subset)

        # Relabel the samples
        relabel_map = {x: idx for idx, x in enumerate(sorted(np.unique(self.labels)))}
        self.labels = [relabel_map[x] for x in self.labels]
        self.labels_txt_to_num = {self.labels_num_to_txt[old_id]: new_id for old_id, new_id in relabel_map.items()}
        self.labels_num_to_txt = {num: text for text, num in self.labels_txt_to_num.items()}

    def _extract_subset(self, num_classes):
        assert num_classes in {100, 200}, "Invalid number of classes!"
        with open(Path(__file__).parent.joinpath(f'imagenet_c/imagenet_{num_classes}_classes.json')) as f:
            subset_class_names = sorted(json.load(f))
            subset_class_numbers = set([self.labels_txt_to_num[x] for x in subset_class_names])
        images, labels = [], []
        for i, l in zip(self.images, self.labels):
            if l in subset_class_numbers:
                images.append(i)
                labels.append(l)
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img = PIL.Image.open(self.images[index]).convert('RGB')
        img = self.transform(img)
        label = self.labels[index]
        return img, label, -1

    def __len__(self):
        return len(self.labels)

    def get_test_dataloader(self, batch_size, num_workers, shuffle=False):
        test_dataloader = DataLoader(self, batch_size, shuffle, num_workers=num_workers, persistent_workers=True)
        return test_dataloader

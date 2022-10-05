import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import PIL.Image
import lmdb
import numpy as np
import pyarrow as pa
import torch
import torchvision.datasets
from PIL import PngImagePlugin
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.transforms import transforms, InterpolationMode

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)


class CIFAR10:
    def __init__(
            self,
            root: str,
            download: bool = True,
    ) -> None:
        self.root = root
        self.download = download
        self._split = [45000, 5000]

        _normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
        dataset = torchvision.datasets.CIFAR10(self.root, True, self._transform_train, None, self.download)
        data_split_train, data_split_val = random_split(dataset, self._split, torch.Generator().manual_seed(99))
        train_loader = DataLoader(data_split_train, batch_size, shuffle, num_workers=num_workers,
                                  persistent_workers=True)
        return train_loader

    def get_validation_dataloader(self, batch_size, num_workers, shuffle=False):
        return self.get_test_dataloader(batch_size, num_workers, shuffle)

    def get_test_dataloader(self, batch_size, num_workers, shuffle=False):
        dataset = torchvision.datasets.CIFAR10(self.root, False, self._transform_test, None, self.download)
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
        return img, label

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
            use_subset: bool,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.use_subset = use_subset

        _normalize = transforms.Normalize((0.485, 0.456, 0.406), (00.229, 0.224, 0.225))
        self._transform_train = transforms.Compose([
            transforms.Resize(256, InterpolationMode.BILINEAR),
            transforms.RandomCrop(img_size),
            # transforms.CenterCrop(img_size),
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

        if self.use_subset:
            self.num_classes = self.use_subset
        else:
            self.num_classes = 1000

        self._setup()
        # self._lmdb_setup()

    def _lmdb_setup(self):

        if self.split == 'train':
            self.transform = self._transform_train
        elif self.split == 'val':
            self.transform = self._transform_test
        self.root = str(self.root.joinpath(self.split))

        self.env = lmdb.open(self.root, subdir=os.path.isdir(self.root),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

    def _setup(self):
        labels_text = sorted([x.stem for x in self.root.glob('Annotations/CLS-LOC/train/*')])
        self.labels_txt_to_num = {x: idx for idx, x in enumerate(labels_text)}
        self.labels_num_to_txt = {idx: x for idx, x in enumerate(labels_text)}

        if self.split == 'train':
            self.data = sorted(self.root.glob('Data/CLS-LOC/train/*/*'))
            self.labels = [self.labels_txt_to_num[x.stem.split('_')[0]] for x in self.data]
            self.transform = self._transform_train

        elif self.split == 'val':
            val_annotations = self.root.glob('Annotations/CLS-LOC/val/*.xml')
            self.labels = {}
            for file in val_annotations:
                root = ET.parse(file).getroot()
                label = root[5][0].text
                filename = root[1].text
                self.labels[filename] = self.labels_txt_to_num[label]
            self.data = sorted(self.root.glob('Data/CLS-LOC/val/*'))
            self.labels = [self.labels[x.stem] for x in self.data]
            self.transform = self._transform_test

        else:
            raise ValueError('Invalid split')

        if self.use_subset:
            self._extract_subset(num_classes=self.use_subset)

        # Relabel the samples
        relabel_map = {x: idx for idx, x in enumerate(sorted(np.unique(self.labels)))}
        self.labels = [relabel_map[x] for x in self.labels]
        self.labels_txt_to_num = {self.labels_num_to_txt[old_id]: new_id for old_id, new_id in relabel_map.items()}
        self.labels_num_to_txt = {num: text for text, num in self.labels_txt_to_num.items()}

        self.length = len(self.labels)

    def _extract_subset(self, num_classes):
        assert num_classes in {100, 200}, 'Invalid number of subset classes!'
        with open(Path(__file__).parent.joinpath(f'imagenet_c/imagenet_{num_classes}_classes.json')) as f:
            subset_class_names = sorted(json.load(f))
            subset_class_numbers = set([self.labels_txt_to_num[x] for x in subset_class_names])
        data, labels = [], []
        for d, l in zip(self.data, self.labels):
            if l in subset_class_numbers:
                data.append(d)
                labels.append(l)
        self.data = data
        self.labels = labels

    def __getitem__(self, item):
        img = PIL.Image.open(self.data[item]).convert('RGB')
        img = self.transform(img)
        label = self.labels[item]
        return img, label

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
    def __init__(self, root: str, img_size: int, use_subset=None):
        self.root = root
        self.img_size = img_size
        self.use_subset = use_subset

    def get_train_dataloader(self, batch_size, num_workers, shuffle=True):
        self.dataset = _ImageNetBase(self.root, 'train', self.img_size, self.use_subset)
        train_loader = DataLoader(self.dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                  prefetch_factor=16, pin_memory=True, persistent_workers=True)
        return train_loader

    def get_validation_dataloader(self, batch_size=None, num_workers=None, shuffle=False):
        self.dataset = _ImageNetBase(self.root, 'val', self.img_size, self.use_subset)
        val_loader = DataLoader(self.dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                prefetch_factor=16, pin_memory=True, persistent_workers=True)
        return val_loader

    def get_test_dataloader(self, batch_size, num_workers, shuffle=False):
        self.dataset = _ImageNetBase(self.root, 'val', self.img_size, self.use_subset)
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
        return img, label

    def __len__(self):
        return len(self.labels)

    def get_test_dataloader(self, batch_size, num_workers, shuffle=False):
        test_dataloader = DataLoader(self, batch_size, shuffle, num_workers=num_workers, persistent_workers=True)
        return test_dataloader

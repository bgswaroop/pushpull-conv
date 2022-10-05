import argparse
import io
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import lmdb
# This segfaults when imported before torch: https://github.com/apache/arrow/issues/2637
import pyarrow as pa
import six
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, InterpolationMode


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        img_buf, target = pa.deserialize(byteflow)

        # load image
        img = Image.open(io.BytesIO(img_buf)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


class TransformImage:
    def __init__(self):
        self.resize = Resize(size=256, interpolation=InterpolationMode.BILINEAR)

    def __call__(self, x):
        x = Image.open(io.BytesIO(x)).convert('RGB')
        x = self.resize(x)
        buff = io.BytesIO()
        x.save(buff, format="PNG")
        x = buff.getvalue()
        return x


class ValDataset:
    def __init__(self, img_paths, labels):
        self.img_paths = img_paths
        self.labels = labels
        self.resize = Resize(size=256, interpolation=InterpolationMode.BILINEAR)

    def __getitem__(self, item):
        x = self.img_paths[item]
        x = Image.open(x).convert('RGB')
        x = self.resize(x)
        buff = io.BytesIO()
        x.save(buff, format="PNG")
        img = buff.getvalue()
        label = self.labels[item]
        return img, label

    def __len__(self):
        return len(self.labels)


def folder2lmdb(dataset_path, lmdb_outpath, split, annotations_path=None, write_frequency=5000):
    directory = os.path.expanduser(dataset_path)
    print("Loading dataset from %s" % directory)

    if split == 'train':
        dataset = ImageFolder(directory, loader=raw_reader, transform=TransformImage())
        # img_paths = Path(dataset_path).glob('*/*')
        # labels = [labels_txt_to_num[x] for x in labels]
        # img_paths = [str(x) for x in img_paths]
        # dataset = ValDataset(img_paths, labels)
    elif split == 'val':
        val_annotations = Path(annotations_path).glob('*.xml')
        labels = dict()
        labels_text = set()
        for file in val_annotations:
            root = ET.parse(file).getroot()
            label = root[5][0].text
            filename = root[1].text
            labels[filename] = label
            labels_text.add(label)
        img_paths = sorted(Path(dataset_path).glob('*'))
        labels = [labels[x.stem] for x in img_paths]
        labels_txt_to_num = {x: idx for idx, x in enumerate(sorted(labels_text))}
        labels = [labels_txt_to_num[x] for x in labels]
        img_paths = [str(x) for x in img_paths]
        dataset = ValDataset(img_paths, labels)
    else:
        raise ValueError('Invalid Split')

    data_loader = DataLoader(dataset, num_workers=32, collate_fn=lambda x: x)
    # print(dataset.classes)

    lmdb_path = os.path.expanduser(lmdb_outpath)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, img_paths in enumerate(data_loader):
        img_in_bytes, label = img_paths[0]

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((img_in_bytes, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def create_lmdb_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Path to original image dataset folder")
    parser.add_argument("-o", "--outpath", help="Path to output LMDB file")
    parser.add_argument("-a", "--annotations", help="Path to ImageNet annotations folder")
    parser.add_argument("-s", "--split", choices=['train', 'val'])
    args = parser.parse_args()
    folder2lmdb(args.dataset, args.outpath, args.split, args.annotations)


if __name__ == "__main__":
    # create_lmdb_dataset()
    dataset = ImageFolderLMDB(db_path='/home/guru/datasets/imagenet/imagenet_lmdb/val')
    item = dataset.__getitem__(0)

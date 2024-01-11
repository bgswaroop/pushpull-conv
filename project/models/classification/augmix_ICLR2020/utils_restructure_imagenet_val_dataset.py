# This code re-stuctures the ImageNet validation dataset to be the same as the train dataset.
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

parser = argparse.ArgumentParser(description='Restructures an Imagenet validation data set')
parser.add_argument('val_data', help='path to original validation data set as downloaded')
parser.add_argument('val_data_annotations', help='path to corresponding annotations of validation data of Imagenet')
args = parser.parse_args()


def restructure_dataset():
    dest_dir = Path(args.val_data).parent.joinpath('val_restructured')
    val_data = sorted(Path(args.val_data).glob('*'))
    annotations = sorted(Path(args.val_data_annotations).glob('*'))

    for img, meta_data in zip(val_data, annotations):
        root = ET.parse(meta_data).getroot()
        label = root[5][0].text
        assert root[1].text == img.name.split('.')[0], 'image name and annotation match error'
        dest_dir.joinpath(label).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dest_dir.joinpath(label))


if __name__ == '__main__':
    restructure_dataset()
    print('Run finished!')

import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import wget
from tqdm import tqdm


def bar_progress(current, total, width=80):
    """
    Credits - https://stackoverflow.com/a/61346454/2709971
    """
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def prepare_imagenet_1k(download: bool = False, dataset_dir: str = None):
    """
    Download and restructure the ImageNet dataset
    Note - Running this method to data does not give the user the licence to re-distribute the dataset.
    Kindly refer to the website https://www.image-net.org/ for terms of download and use.
    This code is provided only to allow replication of the reported results and nothing else.

    :param download: Boolean flag. Defaults to False. Set this to True to download the dataset
    :param dataset_dir: The directory in which to load and process the dataset
    :return:
    """
    temp_directories = ['train_images', 'val_images', 'val_annotations']

    if download:
        train_images_url = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
        val_images_url = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
        val_annotations_url = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz"
        urls = [train_images_url, val_images_url, val_annotations_url]

        assert dataset_dir is not None, "Download location is not specified!"
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)

        for url, target_dir in zip(urls, temp_directories):
            downloaded_file = wget.download(url, dataset_dir, bar=bar_progress)
            download_file_target_dir = Path(dataset_dir).joinpath(target_dir)
            print('\n')  # to flush the progress bar
            print(f'Downloaded the file - {downloaded_file}!')

            with tarfile.open(downloaded_file) as t:
                extract_dir = Path(dataset_dir).joinpath(t.firstmember.name)
                t.extractall(dataset_dir)
                extract_dir.rename(download_file_target_dir)
                print(f'Finished extracting the archive!')

            Path(downloaded_file).unlink(missing_ok=True)
            print(f'Deleted the archive (if present) - {downloaded_file}!')

    # restructure the ImageNet dataset
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir.joinpath('imagenet').joinpath('train')
    val_dir = dataset_dir.joinpath('imagenet').joinpath('val')
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for archive in tqdm(sorted(dataset_dir.glob('train_images/*.tar'))):
        with tarfile.open(archive) as t:
            extract_dir = train_dir.joinpath(archive.stem)
            t.extractall(extract_dir)

    for file in tqdm(sorted(dataset_dir.glob('val_annotations/*.xml'))):
        root = ET.parse(file).getroot()
        label = root[5][0].text
        filename = root[1].text

        image = list(dataset_dir.glob(f'val_images/{filename}*'))[0]
        val_dir.joinpath(label).mkdir(exist_ok=True, parents=True)
        shutil.move(image, val_dir.joinpath(label).joinpath(image.name))

    # Delete the unstructured temp datasets
    for temp_dir in temp_directories:
        shutil.rmtree(Path(dataset_dir).joinpath(temp_dir))


def prepare_imagenet_c(download: bool = False, dataset_dir: str = None):
    """
    Download and restructure the ImageNet dataset
    Note - Running this method to data does not give the user the licence to re-distribute the dataset.
    Kindly refer to the website https://zenodo.org/records/2235448 for terms of download and use.
    This code is provided only to allow replication of the reported results and nothing else.

    :param download: Boolean flag. Defaults to False. Set this to True to download the dataset
    :param dataset_dir: The directory in which to load and process the dataset
    :return:
    """
    datasets = {
        'blur': 'https://zenodo.org/records/2235448/files/blur.tar',
        'digital': 'https://zenodo.org/records/2235448/files/digital.tar',
        'extra': 'https://zenodo.org/records/2235448/files/extra.tar',
        'noise': 'https://zenodo.org/records/2235448/files/noise.tar',
        'weather': 'https://zenodo.org/records/2235448/files/weather.tar',
    }
    if download:
        assert dataset_dir is not None, "Download location is not specified!"
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)

        for dataset_name, url in datasets.items():
            downloaded_file = wget.download(url, dataset_dir, bar=bar_progress)
            print('\n')  # to flush the progress bar
            print(f'Downloaded the file - {downloaded_file}!')

            with tarfile.open(str(Path(dataset_dir).joinpath(f'{dataset_name}.tar'))) as t:
                t.extractall(str(Path(dataset_dir).joinpath(f'{dataset_name}')))
                print(f'Finished extracting the archive!')

            Path(downloaded_file).unlink(missing_ok=True)
            print(f'Deleted the archive (if present) - {downloaded_file}!')

    # restructure the ImageNet-C dataset
    dataset_dir = Path(dataset_dir)
    imagenet_c_dir = dataset_dir.joinpath('imagenet-c')
    imagenet_c_dir.mkdir(parents=True, exist_ok=True)

    for corruption_category in datasets.keys():
        for corruption_type in dataset_dir.joinpath(corruption_category).glob('*'):
            if corruption_type.is_dir():
                shutil.move(corruption_type, imagenet_c_dir)

    # Delete the unstructured temp datasets
    for temp_dir in datasets.keys():
        shutil.rmtree(Path(dataset_dir).joinpath(temp_dir))


if __name__ == '__main__':
    print('Warning - Storage requirements & Compute time\n'
          'Storage requirement - Execution of this code requires about 500 GB of available storage in "download_dir", '
          'to save and process the intermediate data.\n'
          'Computation time - This execution can take several hours to complete - depends upon system\'s '
          'compute capabilities & internet speed\n'
          'If sure, then uncomment the relevant lines of code and set the download parameter to True.')

    # prepare_imagenet_1k(download=False, dataset_dir="/Users/guru/Documents/GitCode/Datasets")
    # prepare_imagenet_c(download=False, dataset_dir="/Users/guru/Documents/GitCode/Datasets")

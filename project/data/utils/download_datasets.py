import sys
import tarfile
from pathlib import Path

import wget


def bar_progress(current, total, width=80):
    """
    Credits - https://stackoverflow.com/a/61346454/2709971
    """
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_imagenet_1k(download=False, download_dir=None):
    """
    Download and restructure the ImageNet dataset on UNIX system

    Note - Running this method to data does not give the user the licence to re-distribute the dataset.
    Kindly refer to the website https://www.image-net.org/ for terms of download and use.
    This code is provided only to allow replication of the reported results and nothing else.
    :return:
    """
    train_images_url = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
    train_annotations_url = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_v2.tar.gz"
    val_images_url = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
    val_annotations_url = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz"
    urls = [train_images_url, train_annotations_url, val_images_url, val_annotations_url]
    url_targets = ['train_images', 'train_annotations', 'val_images', 'val_annotations']

    if download:
        assert download_dir is not None, "Download location is not specified!"
        Path(download_dir).mkdir(parents=True, exist_ok=True)

        for url, target_dir in zip(urls, url_targets):
            downloaded_file = wget.download(url, download_dir, bar=bar_progress)
            download_file_target_dir = Path(download_dir).joinpath(target_dir)
            print('\n')  # to flush the progress bar
            print(f'Downloaded the file - {downloaded_file}!')

            tar = tarfile.open(downloaded_file)
            extract_dir = Path(download_dir).joinpath(tar.firstmember.name)
            tar.extractall(download_dir)
            extract_dir.rename(download_file_target_dir)
            tar.close()
            print(f'Finished extracting the archive!')

            Path(downloaded_file).unlink(missing_ok=True)
            print(f'Deleted the archive (if present) - {downloaded_file}!')

    # restructure the ImageNet dataset
    # todo


def download_imagenet_c():
    pass


if __name__ == '__main__':
    print('Warning - Execution of this code requires about 500 GB of available storage in "download_dir", '
          'to save and process the intermediate data. If sure, then uncomment the relevant lines of code.')

    download_imagenet_1k(download_dir="/scratch/p288722/trials")
    download_imagenet_c()

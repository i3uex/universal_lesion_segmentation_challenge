import os

import monai
from config.constants import SPACING, ZENODO_COVID_CASES_PATH, ZENODO_INFECTION_MASKS_PATH, STUDIES_COVID_CASES_PATH, \
    DATASET_VOLUMES_WITH_MASK_RANGE, STUDIES_INFECTION_MASKS_PATH
from utils.helpers import load_images_from_path, get_volumes_in_range
import numpy as np

PATH = "Datasets/preprocessed_2/"

class SpacingDataset(monai.data.PersistentDataset):
    def __init__(self, volumes, transform=None):
        super().__init__(data=volumes, transform=None, cache_dir="covid_dataset_cache")
        self.volumes = volumes
        self.transform = transform

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        volume = self.volumes[index]
        return self.transform(volume)


def get_spacing_transforms():
    return monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=('img', 'mask'), image_only=True, ensure_channel_first=True),
        monai.transforms.Spacingd(keys=('img', 'mask'), pixdim=SPACING, mode=("bilinear", "nearest")),
    ])


def store_volume_to_nifti(volume, path, volume_name):
    img = volume["img"]
    mask = volume["mask"]

    img_writer = monai.data.NibabelWriter()
    mask_writer = monai.data.NibabelWriter()

    print("Storing volume to nifti: ", volume_name)
    print("Image shape: ", img.shape)
    with open(path + "/shapes.txt", "a") as f:
        f.write(volume_name + ": " + str(img.shape) + "\n")

    img_writer.set_data_array(img, channel_dim=0)
    img_writer.write(path + "images/" + volume_name, verbose=True)

    mask_writer.set_data_array(mask, channel_dim=0)
    mask_writer.write(path + "labels/" + volume_name, verbose=True)


def main():
    images_zenodo = load_images_from_path(ZENODO_COVID_CASES_PATH)
    labels_zenodo = load_images_from_path(ZENODO_INFECTION_MASKS_PATH)

    images_studies = get_volumes_in_range(STUDIES_COVID_CASES_PATH, DATASET_VOLUMES_WITH_MASK_RANGE)
    labels_studies = load_images_from_path(STUDIES_INFECTION_MASKS_PATH)

    images = images_zenodo + images_studies
    labels = labels_zenodo + labels_studies

    data_dicts_zenodo = np.array([{"img": img, "mask": mask} for img, mask in zip(images, labels)])

    if not os.path.exists(PATH):
        os.mkdir(PATH)
    if not os.path.exists(PATH + "images"):
        os.mkdir(PATH + "images")
    if not os.path.exists(PATH + "labels"):
        os.mkdir(PATH + "labels")

    for i in range(len(data_dicts_zenodo)):
        dataset = SpacingDataset(volumes=[data_dicts_zenodo[i]], transform=get_spacing_transforms())
        volume_name = data_dicts_zenodo[i]["img"].split("/")[-1]
        store_volume_to_nifti(dataset[0], PATH, volume_name)


if __name__ == '__main__':
    main()
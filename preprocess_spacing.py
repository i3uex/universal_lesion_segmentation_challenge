import os

import monai
from config.constants import SPACING, COVID_CASES_PATH, INFECTION_MASKS_PATH
from utils.helpers import load_images_from_path
import numpy as np

PATH = "Datasets/preprocessed/"

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

    img_writer.set_data_array(img, channel_dim=0)
    img_writer.write(path + "images/" + volume_name, verbose=True)

    mask_writer.set_data_array(mask, channel_dim=0)
    mask_writer.write(path + "labels/" + volume_name, verbose=True)


def main():
    images = load_images_from_path(COVID_CASES_PATH)
    labels = load_images_from_path(INFECTION_MASKS_PATH)

    data_dicts = np.array([{"img": img, "mask": mask} for img, mask in zip(images, labels)])

    if not os.path.exists(PATH):
        os.mkdir(PATH)
    if not os.path.exists(PATH + "images"):
        os.mkdir(PATH + "images")
    if not os.path.exists(PATH + "labels"):
        os.mkdir(PATH + "labels")

    for i in range(len(data_dicts)):
        dataset = SpacingDataset(volumes=[data_dicts[i]], transform=get_spacing_transforms())
        volume_name = data_dicts[i]["img"].split("/")[-1]
        store_volume_to_nifti(dataset[0], PATH, volume_name)


if __name__ == '__main__':
    main()
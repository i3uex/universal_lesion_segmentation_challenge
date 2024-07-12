import torch
import monai

class CovidDataset(monai.data.PersistentDataset):
    def __init__(self, volumes, hrct_transform=None, cbct_transform=None):
        super().__init__(data=volumes, transform=None, cache_dir="covid_dataset_cache")
        self.volumes = volumes
        self.hrct_transform = hrct_transform
        self.cbct_transform = cbct_transform

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        volume = self.volumes[index]

        if "coronacases" in volume["img"]:
            return self.hrct_transform(volume)
        else:
            return self.cbct_transform(volume)


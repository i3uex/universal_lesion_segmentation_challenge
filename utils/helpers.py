from pathlib import Path
import logging
import numpy as np

def load_images_from_path(path: str) -> list[str]:
    """
    This function loads all nii.gz files from a given directory path.

    Args:
        path (str): The directory path from which to load the nii.gz files.

    Returns:
        list[str]: A list of file paths for each nii.gz file in the specified directory.

    Note:
        This function uses Python's built-in pathlib library to iterate over the directory contents.
        It checks each file to ensure it is a file (not a directory) and that it has a .gz extension.
    """
    logging.debug(f"Loading images from {path}")
    return sorted([str(f) for f in Path(path).iterdir() if f.is_file() and f.suffix == '.gz'])


def check_dataset(dataset):
    """
    This function checks the dataset for two conditions:
    1. If the mask has only zeros, it raises a ValueError.
    2. If there are no 'radiopaedia' or 'coronacases' in the dataset, it raises a ValueError.

    Args:
        dataset (list): The dataset to check. Each element in the dataset is a dictionary with keys 'mask' and 'img'.

    Raises:
        ValueError: If the mask has only zeros or if there are no 'radiopaedia' or 'coronacases' in the dataset.

    Returns:
        None
    """
    radiopaedia = False
    coronacases = False
    for i, data in enumerate(dataset):
        if np.unique(data["mask"]).size == 1 and np.unique(data["mask"])[0] == 0:
            raise ValueError("The mask has only zeros")

        if "radiopaedia" in dataset.volumes[i]["img"]:
            radiopaedia = True
        if "coronacases" in dataset.volumes[i]["img"]:
            coronacases = True

    if not radiopaedia or not coronacases:
        raise ValueError("There are no radiopaedia or coronacases in the dataset")

    return


def ckpt_path(version: int) -> str:
    """
    This function generates the path to the checkpoint file for a given version of the model.

    Args:
        version (int): The version number of the model.

    Returns:
        str: The path to the checkpoint file.

    Note:
        This function uses Python's built-in pathlib library to iterate over the directory contents.
        It checks each file to ensure it is a file (not a directory) and that it has a .ckpt extension.
        The function assumes that the checkpoint files are stored in a directory structure like this:
        `lightning_logs/lightning_logs/version_{version}/checkpoints/`
    """
    return [str(f) for f in Path(f"lightning_logs/lightning_logs/version_{version}/checkpoints/").iterdir() if f.is_file() and f.suffix == '.ckpt'][0]

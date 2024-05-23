from pathlib import Path
import logging


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
    return [str(f) for f in Path(path).iterdir() if f.is_file() and f.suffix == '.gz']
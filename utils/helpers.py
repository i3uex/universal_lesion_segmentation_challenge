from pathlib import Path
import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from skimage import measure


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


def get_volumes_in_range(images_path: str, mask_range: tuple[str, str]):
    images = []
    for subdir in Path(images_path).iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.is_file() and mask_range[0] <= f.stem.split("_")[1].split(".")[0] <= mask_range[1]:
                    images.append(str(f))
    return sorted(images)


def burn_masks_in_ct(ct, mask, predictions, path_to_save=None):
    # Definir los valores de las clases
    BACKGROUND = 0
    CLASS1 = 1

    # Cambiar los valores de la matriz "predictions_without_black" a los valores de las clases
    predictions_without_black = np.where(predictions == BACKGROUND, np.nan, CLASS1)

    # Crear el mapa de colores personalizado
    cmap = ListedColormap(['yellow', 'black'])

    # Crear la figura y los ejes
    dpi = 80
    figsize = 512 / dpi

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(figsize + 1, figsize + 1.2), dpi=dpi)

    # Mostrar la imagen con el mapa de colores personalizado
    ax.imshow(ct, cmap="gray")
    ax.imshow(predictions_without_black, alpha=0.6, cmap=cmap)

    # Encontrar los contornos de la máscara
    contours = measure.find_contours(mask)

    # Dibujar los contornos
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="blue")

    # Añadir el título
    # ax.set_title("Dice score: {:.3%}".format(dice_score), fontsize=16)

    # Crear la leyenda con texto y colores
    class_patches = [Patch(color='yellow', label='Prediction')]
    line_patches = [Patch(color='blue', label='Mask contour')]

    legend = ax.legend(handles=class_patches + line_patches, loc='upper left')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.8)  # Agregar transparencia al marco
    legend.get_texts()[0].set_fontsize(8)  # Ajustar el tamaño del texto
    legend.get_texts()[1].set_fontsize(8)

    # Quitar los ejes xticks e yticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Guardar la figura
    plt.savefig(path_to_save, bbox_inches='tight')

    # Cerrar la figura
    plt.close()

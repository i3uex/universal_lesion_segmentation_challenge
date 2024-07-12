COVID_CASES_PATH = "Datasets/Zenodo/COVID-19-CT-Seg_20cases/"
INFECTION_MASKS_PATH = "Datasets/Zenodo/Infection_Mask/"
LUNG_MASKS_PATH = "Datasets/Zenodo/Lung_Mask/"
LUNG_AND_INFECTION_MASKS_PATH = "Datasets/Zenodo/Lung_and_Infection_Mask/"

COVID_PREPROCESSED_CASES_PATH = "Datasets/preprocessed/images/"
INFECTION_PREPROCESSED_MASKS_PATH = "Datasets/preprocessed/labels/"

EXPERIMENTS_PATH = "Experiments/"
SEED = 420
IMG_SIZE = (512, 512)
VALIDATION_INFERENCE_ROI_SIZE = (64,) * 3
SPATIAL_SIZE = (64,) * 3  # In case of changing it, remember to take into account the padding in the transforms
NUM_RAND_PATCHES = 16
LEVEL = -650
WIDTH = 1500
SPACING = (0.6836,)*3
LOWER_BOUND_WINDOW_HRCT = LEVEL - (WIDTH // 2)
UPPER_BOUND_WINDOW_HRCT = LEVEL + (WIDTH // 2)
LOWER_BOUND_WINDOW_CBCT = 0.0
UPPER_BOUND_WINDOW_CBCT = 255.0
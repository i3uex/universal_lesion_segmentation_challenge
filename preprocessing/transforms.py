import monai
from config.constants import LOWER_BOUND_WINDOW_HRCT, UPPER_BOUND_WINDOW_HRCT, LOWER_BOUND_WINDOW_CBCT, \
    UPPER_BOUND_WINDOW_CBCT, SPATIAL_SIZE, NUM_RAND_PATCHES, IMG_SIZE, SPACING


def get_hrct_transforms():
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=('img', 'mask'), image_only=True, ensure_channel_first=True),
            monai.transforms.ThresholdIntensityd(keys=("img",), threshold=LOWER_BOUND_WINDOW_HRCT, above=True,
                                                 cval=LOWER_BOUND_WINDOW_HRCT),
            monai.transforms.ThresholdIntensityd(keys=("img",), threshold=UPPER_BOUND_WINDOW_HRCT, above=False,
                                                 cval=UPPER_BOUND_WINDOW_HRCT),
            monai.transforms.ScaleIntensityd(keys='img', minv=0.0, maxv=1.0),
            monai.transforms.CropForegroundd(keys=["img", "mask"], source_key="img"),
            monai.transforms.Orientationd(keys=('img', 'mask'), axcodes="PLI"),
            monai.transforms.RandCropByPosNegLabeld(keys=('img', 'mask'), label_key="mask",
                                                    spatial_size=SPATIAL_SIZE, pos=1, neg=1,
                                                    num_samples=NUM_RAND_PATCHES, allow_smaller=True),
            monai.transforms.SpatialPadd(keys=('img', 'mask'), spatial_size=SPATIAL_SIZE, method="symmetric"),

            monai.transforms.RandFlipd(keys=('img', 'mask'), prob=0.2, spatial_axis=0),
            monai.transforms.RandFlipd(keys=('img', 'mask'), prob=0.2, spatial_axis=1),
            monai.transforms.RandFlipd(keys=('img', 'mask'), prob=0.2, spatial_axis=2),
            monai.transforms.RandZoomd(keys=('img', 'mask'), prob=0.4, min_zoom=0.9, max_zoom=1.1),

            monai.transforms.ToTensord(keys=("img", "mask")),
        ]
    )


def get_cbct_transforms():
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=('img', 'mask'), image_only=True, ensure_channel_first=True),
            monai.transforms.ScaleIntensityd(keys='img', minv=0.0, maxv=1.0),
            monai.transforms.CropForegroundd(keys=["img", "mask"], source_key="img"),
            monai.transforms.Orientationd(keys=('img', 'mask'), axcodes="ALI"),
            monai.transforms.RandCropByPosNegLabeld(keys=('img', 'mask'), label_key="mask",
                                                    spatial_size=SPATIAL_SIZE, pos=1, neg=1,
                                                    num_samples=NUM_RAND_PATCHES, allow_smaller=True),
            monai.transforms.SpatialPadd(keys=('img', 'mask'), spatial_size=SPATIAL_SIZE, method="symmetric"),

            monai.transforms.RandFlipd(keys=('img', 'mask'), prob=0.2, spatial_axis=0),
            monai.transforms.RandFlipd(keys=('img', 'mask'), prob=0.2, spatial_axis=1),
            monai.transforms.RandFlipd(keys=('img', 'mask'), prob=0.2, spatial_axis=2),
            monai.transforms.RandZoomd(keys=('img', 'mask'), prob=0.4, min_zoom=0.9, max_zoom=1.1),

            monai.transforms.ToTensord(keys=("img", "mask")),
        ]
    )


def get_val_hrct_transforms():
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=('img', 'mask'), image_only=True, ensure_channel_first=True),
            monai.transforms.ThresholdIntensityd(keys=("img",), threshold=LOWER_BOUND_WINDOW_HRCT, above=True,
                                                 cval=LOWER_BOUND_WINDOW_HRCT),
            monai.transforms.ThresholdIntensityd(keys=("img",), threshold=UPPER_BOUND_WINDOW_HRCT, above=False,
                                                 cval=UPPER_BOUND_WINDOW_HRCT),
            monai.transforms.ScaleIntensityd(keys='img', minv=0.0, maxv=1.0),
            monai.transforms.CropForegroundd(keys=["img", "mask"], source_key="img"),
            monai.transforms.Orientationd(keys=('img', 'mask'), axcodes="PLI"),
            monai.transforms.ToTensord(keys=("img", "mask")),
        ]
    )


def get_val_cbct_transforms():
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=('img', 'mask'), image_only=True, ensure_channel_first=True),
            monai.transforms.ScaleIntensityd(keys='img', minv=0.0, maxv=1.0),
            monai.transforms.CropForegroundd(keys=["img", "mask"], source_key="img"),
            monai.transforms.Orientationd(keys=('img', 'mask'), axcodes="ALI"),
            monai.transforms.ToTensord(keys=("img", "mask")),
        ]
    )
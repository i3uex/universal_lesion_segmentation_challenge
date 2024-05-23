import monai

from config.constants import LOWER_BOUND_WINDOW_HRCT, UPPER_BOUND_WINDOW_HRCT, LOWER_BOUND_WINDOW_CBCT, \
    UPPER_BOUND_WINDOW_CBCT


def get_hrct_transforms():
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=('img', 'mask'), image_only=True, ensure_channel_first=True),
            monai.transforms.Orientationd(keys=('img', 'mask'), axcodes="ALI"),
            monai.transforms.RandCropByPosNegLabeld(keys=('img', 'mask'), label_key="mask",
                                                    spatial_size=(96, 96, 96), pos=1, neg=1,
                                                    num_samples=4, allow_smaller=True),
            monai.transforms.SpatialPadd(keys=('img', 'mask'), spatial_size=(96, 96, 96), method='symmetric'), # TODO check if end or symmetric
            monai.transforms.ScaleIntensityRangeD(keys='img', a_min=LOWER_BOUND_WINDOW_HRCT,
                                                  a_max=UPPER_BOUND_WINDOW_HRCT, b_min=0.0, b_max=1.0, clip=True),
            monai.transforms.ToTensord(keys=("img", "mask")),
        ]
    )


def get_cbct_transforms():
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=('img', 'mask'), image_only=True, ensure_channel_first=True),
            monai.transforms.Orientationd(keys=('img', 'mask'), axcodes="ALI"),
            monai.transforms.RandCropByPosNegLabeld(keys=('img', 'mask'), label_key="mask",
                                                    spatial_size=(96, 96, 96), pos=1, neg=1,
                                                    num_samples=4, allow_smaller=True),
            monai.transforms.SpatialPadd(keys=('img', 'mask'), spatial_size=(96, 96, 96), method='symmetric'), # TODO check if end or symmetric
            monai.transforms.ScaleIntensityd(keys='img', minv=LOWER_BOUND_WINDOW_CBCT, maxv=UPPER_BOUND_WINDOW_CBCT),
            monai.transforms.ToTensord(keys=("img", "mask")),
        ]
    )


def get_val_hrct_transforms():
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=('img', 'mask'), image_only=True, ensure_channel_first=True),
            monai.transforms.Orientationd(keys=('img', 'mask'), axcodes="ALI"),
            monai.transforms.ScaleIntensityRangeD(keys='img', a_min=LOWER_BOUND_WINDOW_HRCT,
                                                  a_max=UPPER_BOUND_WINDOW_HRCT, b_min=0.0, b_max=1.0,
                                                  clip=True),
            monai.transforms.ToTensord(keys=("img", "mask")),
        ]
    )


def get_val_cbct_transforms():
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=('img', 'mask'), image_only=True, ensure_channel_first=True),
            monai.transforms.Orientationd(keys=('img', 'mask'), axcodes="ALI"),
            monai.transforms.ScaleIntensityd(keys='img', minv=LOWER_BOUND_WINDOW_CBCT, maxv=UPPER_BOUND_WINDOW_CBCT),
            monai.transforms.ToTensord(keys=("img", "mask")),
        ]
    )
from monai.networks.nets import UNet, UNETR, SwinUNETR
from config.constants import SPATIAL_SIZE

covid_unet = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
    #dropout=0.3
)

covid_unetr = UNETR(
    in_channels=1,
    out_channels=1,
    img_size=SPATIAL_SIZE,
    feature_size=12,
    hidden_size=768,
    mlp_dim=2048,
    num_heads=8,
    proj_type='conv',
    norm_name='instance',
    res_block=True,
    #dropout_rate=0.3
)

covid_swin_unet = SwinUNETR(
    in_channels=1,
    out_channels=1,
    img_size=SPATIAL_SIZE,
    feature_size=12,
    #drop_rate=0.3,
    norm_name="instance"
)
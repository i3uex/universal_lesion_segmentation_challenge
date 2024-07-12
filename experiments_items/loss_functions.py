import monai
import torch

cross_entropy_loss = torch.nn.BCEWithLogitsLoss()
dice_loss = monai.losses.DiceLoss(sigmoid=True)
haussdorf_loss = monai.losses.HausdorffDTLoss(sigmoid=True)
generalize_dice_focal_loss = monai.losses.GeneralizedDiceFocalLoss(sigmoid=True)
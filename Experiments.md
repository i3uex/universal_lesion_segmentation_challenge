# Experiment
## Loss Functions
- Distribution-based loss (torch.nn.BCEWithLogitsLoss)
- Region-based loss (DiceLoss)
- Distance-based loss (HausdorffDTLoss)
- Compund-based loss (GeneralizedDiceFocalLoss, DiceCE)

## Architectures
- U-Net
- UNETR
- Swin-UNETR

## Metrics
- DSC
- NSD (Surface Dice in monai)
- IoU
- HD (percentile=95.)

## Dump format
The dump will be in CSV files in a folder called `Experiments` with the following structure:
Experiments
- NameArch_LossFunction_1
  - checkpoint: checkpoint del modelo
  - train.csv: epoch | train_loss | train_metrics | val_loss | val_metrics 
  - test.csv: volume | test_loss | test_metric

There will be a total of 12 experiments: `3 architectures x 4 loss functions`

We will choose the best model based on the validation metric. 

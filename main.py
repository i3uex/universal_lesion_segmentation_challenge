import logging
import argparse
import os
import pandas as pd

from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from matplotlib.colors import ListedColormap
from monai.inferers import sliding_window_inference
from skimage import measure

import experiments_items.nets
from monai.metrics.meandice import compute_dice
from config.config import Config
from preprocessing.covid_dataset import CovidDataset
import monai.data
from monai.data import DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from preprocessing.transforms import get_hrct_transforms, get_cbct_transforms, \
    get_val_hrct_transforms, get_val_cbct_transforms
from utils.custom_callbacks import CustomTimingCallback
from utils.helpers import load_images_from_path, check_dataset, burn_masks_in_ct
from config.constants import (ZENODO_COVID_CASES_PATH, ZENODO_INFECTION_MASKS_PATH, SEED, VALIDATION_INFERENCE_ROI_SIZE,
                              SPATIAL_SIZE,
                              ZENODO_LUNG_MASKS_PATH, EXPERIMENTS_PATH, COVID_PREPROCESSED_CASES_PATH,
                              INFECTION_PREPROCESSED_MASKS_PATH, SPACING)
import torch
import numpy as np
from monai.metrics import DiceMetric
import lightning as L
import matplotlib.pyplot as plt

# torch.set_float32_matmul_precision('high')


class Net(L.pytorch.LightningModule):
    def __init__(self, learning_rate: float, model: torch.nn.Module, loss_function: torch.nn, volumes_path: str,
                 masks_path: str, experiment_name: str):
        super(Net, self).__init__()

        # volumes paths
        self.volumes_path = volumes_path
        self.masks_path = masks_path

        # Model, loss function and learning rate
        self.model = model
        print(f"Using model: {type(self.model)}")
        self.loss_function = loss_function
        print(f"Using loss: {type(self.loss_function)}")
        self.learning_rate = learning_rate
        print(f"Using lr: {learning_rate}")

        self.experiment_name = experiment_name

        self.save_hyperparameters(ignore=["model", "loss_function", "volumes_path", "masks_path", "experiment_name"])

        # Define the post-processing transforms
        self.post_pred = monai.transforms.Compose(
            [monai.transforms.EnsureType(data_type='tensor'), monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)])
        self.post_label = monai.transforms.Compose([monai.transforms.AsDiscrete(threshold=0.5)])

        # Dice metric
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.train_dice_metric = DiceMetric(include_background=True, reduction="mean")

        # Surface dice metric
        self.surface_dice_metric = monai.metrics.SurfaceDiceMetric(include_background=True, distance_metric="euclidean", class_thresholds=[1.0])
        self.train_surface_dice_metric = monai.metrics.SurfaceDiceMetric(include_background=True, distance_metric="euclidean", class_thresholds=[1.0])

        # Haussdorf metric
        self.haussdorf_metric = monai.metrics.HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=95)
        self.train_haussdorf_metric = monai.metrics.HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=95)

        # IoU metric
        self.iou_metric = monai.metrics.MeanIoU(include_background=True)
        self.train_iou_metric = monai.metrics.MeanIoU(include_background=True)

        # Best validation dice and epoch
        self.best_val_dice = 0
        self.best_val_epoch = 0

        # Losses lists
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.test_step_outputs = []

        # CSV data frame for metrics
        self.train_val_dump_data_frame = []
        self.test_dump_data_frame = []

        # Paths lists for datasets
        self.test_paths = None
        self.val_paths = None
        self.train_paths = None

        # Datasets
        self.training_ds = None
        self.validation_ds = None
        self.test_ds = None

    def forward(self, x):
        return self.model(x)

    def prepare_data(self) -> None:
        # Load images and masks
        logging.info(f"Loading images from {self.volumes_path} and masks from {self.masks_path}")
        images = load_images_from_path(self.volumes_path)
        labels = load_images_from_path(self.masks_path)

        # Take only the images that are from Mosmed
        train_images = [image for image in images if "radiopaedia" not in image and "coronacases" not in image]
        train_labels = [label for label in labels if "radiopaedia" not in label and "coronacases" not in label]

        # Convert images and masks to a list of dictionaries with keys "img" and "mask"
        data_train_dicts = np.array([{"img": img, "mask": mask} for img, mask in zip(train_images, train_labels)])
        logging.debug(data_train_dicts)

        # Shuffle the data
        shuffler = np.random.RandomState(SEED)
        shuffler.shuffle(data_train_dicts)
        data_train_dicts = list(data_train_dicts)

        # Split the training data into training and validation
        val_split = int(len(data_train_dicts) * 0.2)

        self.train_paths = data_train_dicts[val_split:]
        self.val_paths = data_train_dicts[:val_split]

        # Take coronacases and radiopeadia images for testing
        test_images = [image for image in images if "radiopaedia" in image or "coronacases" in image]
        test_labels = [label for label in labels if "radiopaedia" in label or "coronacases" in label]
        self.test_paths = np.array([{"img": img, "mask": mask} for img, mask in zip(test_images, test_labels)])

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            # Define the CovidDataset instances for training, validation, and test
            self.training_ds = CovidDataset(volumes=self.train_paths, hrct_transform=get_hrct_transforms(),
                                            cbct_transform=get_cbct_transforms())
            self.validation_ds = CovidDataset(volumes=self.val_paths, hrct_transform=get_val_hrct_transforms(),
                                              cbct_transform=get_val_cbct_transforms())
            # Check the dataset
            print("Checking the validation dataset")
            # check_dataset(self.validation_ds)

        if stage == "validate" or stage is None:
            self.validation_ds = CovidDataset(volumes=self.val_paths, hrct_transform=get_val_hrct_transforms(),
                                              cbct_transform=get_val_cbct_transforms())
            # Check the dataset
            print("Checking the validation dataset")
            # check_dataset(self.validation_ds)

        if stage == "test" or stage is None:
            self.test_ds = CovidDataset(volumes=self.test_paths, hrct_transform=get_val_hrct_transforms(),
                                        cbct_transform=get_val_cbct_transforms())
            # Check the dataset
            print("Checking the test dataset")
            # check_dataset(self.test_ds)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.training_ds, batch_size=1, shuffle=True, num_workers=4)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.validation_ds, batch_size=1, num_workers=4)
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_ds, batch_size=1, num_workers=4)
        return test_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["mask"]

        # Forward pass
        raw_outputs = self.forward(inputs)
        loss = self.loss_function(raw_outputs, labels)
        self.log("ts_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        outputs = [self.post_pred(i) for i in decollate_batch(raw_outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        # Dice metric
        self.train_dice_metric(y_pred=outputs, y=labels)
        self.log("ts_dice", self.train_dice_metric.aggregate().item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)

        # NSDice Metric
        self.train_surface_dice_metric(y_pred=outputs, y=labels, spacing=SPACING)
        self.log("ts_surface_dice", self.train_surface_dice_metric.aggregate().item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)

        # Haussdorf Metric
        self.train_haussdorf_metric(y_pred=outputs, y=labels, spacing=SPACING)
        self.log("ts_hd95", self.train_haussdorf_metric.aggregate().item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)

        # IoU Metric
        self.train_iou_metric(y_pred=outputs, y=labels)
        self.log("ts_iou", self.train_iou_metric.aggregate().item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)

        # Store the loss
        train_loss_dictionary = {"loss": loss}
        self.train_step_outputs.append(train_loss_dictionary)

        return loss

    def on_train_epoch_end(self) -> None:
        # Loss
        avg_loss = torch.stack([i["loss"] for i in self.train_step_outputs]).mean()
        self.train_step_outputs.clear()

        # Dice
        mean_train_dice = self.train_dice_metric.aggregate().item()
        self.train_dice_metric.reset()

        # NSDice
        mean_train_surface_dice = self.train_surface_dice_metric.aggregate().item()
        self.train_surface_dice_metric.reset()

        # Haussdorf
        mean_train_haussdorf = self.train_haussdorf_metric.aggregate().item()
        self.train_haussdorf_metric.reset()

        # IoU
        mean_train_iou = self.train_iou_metric.aggregate().item()
        self.train_iou_metric.reset()

        # Log the metrics
        self.log_dict({
            "train_loss": avg_loss,
            "train_dice": mean_train_dice,
            "train_surface_dice": mean_train_surface_dice,
            "train_haussdorf": mean_train_haussdorf,
            "train_iou": mean_train_iou
        }, on_epoch=True, on_step=False, prog_bar=True)

        # Save the metrics to a pandas dataframe
        self.train_val_dump_data_frame[-1].update({
            "train_loss": avg_loss.item(),
            "train_dice": mean_train_dice,
            "train_surface_dice": mean_train_surface_dice,
            "train_haussdorf": mean_train_haussdorf,
            "train_iou": mean_train_iou
        })

        # Log the metrics to tensorboard
        self.logger.experiment.add_scalars("losses", {"train": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("dice", {"train": mean_train_dice}, self.current_epoch)
        self.logger.experiment.add_scalars("surface_dice", {"train": mean_train_surface_dice}, self.current_epoch)
        self.logger.experiment.add_scalars("haussdorf", {"train": mean_train_haussdorf}, self.current_epoch)
        self.logger.experiment.add_scalars("iou", {"train": mean_train_iou}, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["mask"]

        # Inference
        roi_size = VALIDATION_INFERENCE_ROI_SIZE
        sw_batch_size = 4
        outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, self.forward, overlap=0.6)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.validation_step_outputs.append({"val_loss": loss})

        # Validation metrics
        self.dice_metric(y_pred=outputs, y=labels)
        self.surface_dice_metric(y_pred=outputs, y=labels, spacing=SPACING)
        self.haussdorf_metric(y_pred=outputs, y=labels, spacing=SPACING)
        self.iou_metric(y_pred=outputs, y=labels)

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()

        # Dice
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        # NSDice
        mean_val_surface_dice = self.surface_dice_metric.aggregate().item()
        self.surface_dice_metric.reset()

        # Haussdorf
        mean_val_haussdorf = self.haussdorf_metric.aggregate().item()
        self.haussdorf_metric.reset()

        # IoU
        mean_val_iou = self.iou_metric.aggregate().item()
        self.iou_metric.reset()

        # Log the metrics
        self.log_dict({
            "val_loss": avg_loss.item(),
            "val_dice": mean_val_dice,
            "val_surface_dice": mean_val_surface_dice,
            "val_haussdorf": mean_val_haussdorf,
            "val_iou": mean_val_iou
        }, prog_bar=True, on_epoch=True, on_step=False)

        # Save the metrics to a pandas dataframe
        self.train_val_dump_data_frame.append({
            "epoch": self.current_epoch,
            "val_loss": avg_loss.item(),
            "val_dice": mean_val_dice,
            "val_surface_dice": mean_val_surface_dice,
            "val_haussdorf": mean_val_haussdorf,
            "val_iou": mean_val_iou
        })

        # Log the metrics to tensorboard
        self.logger.experiment.add_scalars("losses", {"val_loss": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("dice", {"val_dice": mean_val_dice}, self.current_epoch)
        self.logger.experiment.add_scalars("surface_dice", {"val_surface_dice": mean_val_surface_dice}, self.current_epoch)
        self.logger.experiment.add_scalars("haussdorf", {"val_haussdorf": mean_val_haussdorf}, self.current_epoch)
        self.logger.experiment.add_scalars("iou", {"val_iou": mean_val_iou}, self.current_epoch)

        # Save the best model
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["mask"]
        roi_size = VALIDATION_INFERENCE_ROI_SIZE
        sw_batch_size = 4

        outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, self.forward, overlap=0.6)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        # Dice metric
        self.dice_metric(y_pred=outputs, y=labels)
        volume_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        # Surface dice metric
        self.surface_dice_metric(y_pred=outputs, y=labels, spacing=SPACING)
        volume_surface_dice = self.surface_dice_metric.aggregate().item()
        self.surface_dice_metric.reset()

        # Haussdorf metric
        self.haussdorf_metric(y_pred=outputs, y=labels, spacing=SPACING)
        volume_haussdorf = self.haussdorf_metric.aggregate().item()
        self.haussdorf_metric.reset()

        # IoU metric
        self.iou_metric(y_pred=outputs, y=labels)
        volume_iou = self.iou_metric.aggregate().item()
        self.iou_metric.reset()

        # Create a pandas dataframe with batch_idx, test_loss, and test_metric columns
        df = pd.DataFrame({
            "volume": [batch_idx],
            "test_loss": [loss.item()],
            "test_dice": [volume_dice],
            "test_surface_dice": [volume_surface_dice],
            "test_haussdorf": [volume_haussdorf],
            "test_iou": [volume_iou]
        })
        self.test_dump_data_frame.append(df)
        self.test_step_outputs.append({"test_loss": loss})

        if not os.path.exists(EXPERIMENTS_PATH + self.experiment_name + "/test_images"):
            os.mkdir(EXPERIMENTS_PATH + self.experiment_name + "/test_images")

        if not os.path.exists(EXPERIMENTS_PATH + self.experiment_name + "/test_images/" + self.test_ds.volumes[batch_idx]["img"].split("/")[-1].replace(".nii.gz", "")):
            os.mkdir(EXPERIMENTS_PATH + self.experiment_name + "/test_images/" + self.test_ds.volumes[batch_idx]["img"].split("/")[-1].replace(".nii.gz", ""))

        if not os.path.exists(EXPERIMENTS_PATH + self.experiment_name + "/test_predictions"):
            os.mkdir(EXPERIMENTS_PATH + self.experiment_name + "/test_predictions")

        output_writer = monai.data.NibabelWriter()
        output_writer.set_data_array(outputs[0], channel_dim=0)
        output_writer.write(EXPERIMENTS_PATH + self.experiment_name + "/test_predictions/" + self.test_ds.volumes[batch_idx]["img"].split("/")[-1].replace(".nii.gz", "") + "_prediction")

        for j in range(outputs[0].shape[-1]):
            burn_masks_in_ct(inputs[0, 0, :, :, j].cpu().numpy(), labels[0][0, :, :, j].cpu().numpy(),
                             outputs[0][0, :, :, j].cpu().numpy(),
                             path_to_save=f"{EXPERIMENTS_PATH}{self.experiment_name}/test_images/{self.test_ds.volumes[batch_idx]['img'].split('/')[-1].replace('.nii.gz', '')}/slice_{j}.png")

    def on_test_epoch_end(self):
        df = pd.concat(self.test_dump_data_frame)
        df.to_csv(EXPERIMENTS_PATH + self.experiment_name + "/test_metrics.csv", header=True, index=False)

    def on_train_end(self) -> None:
        df = pd.DataFrame(self.train_val_dump_data_frame)
        df.to_csv(EXPERIMENTS_PATH + self.experiment_name + "/train_val_metrics.csv", header=True, index=False)

def main():
    # Initialize the configuration
    config = Config()
    config.set_logging_verbose()

    # Set the seed for reproducibility
    L.seed_everything(SEED)
    torch.manual_seed(SEED)
    monai.utils.set_determinism(seed=SEED)

    # Experiment name
    experiment_name = f"{config.get_network_name()}_{config.get_loss_name()}"
    print(f"Experiment name: {experiment_name}")

    net = Net(
        learning_rate=config.get_learning_rate(),
        model=config.get_network(),
        loss_function=config.get_loss(),
        volumes_path=COVID_PREPROCESSED_CASES_PATH,
        masks_path=INFECTION_PREPROCESSED_MASKS_PATH,
        experiment_name=experiment_name
    )

    tensorboard_logger = L.pytorch.loggers.TensorBoardLogger(save_dir=EXPERIMENTS_PATH, name=experiment_name, version="tensorboard",)

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=EXPERIMENTS_PATH + experiment_name + "/checkpoints",
            monitor="val_dice",
            save_top_k=1,
            mode="max",
            save_last=True
        ),
        CustomTimingCallback()
    ]

    trainer = L.pytorch.Trainer(
        default_root_dir=EXPERIMENTS_PATH,
        devices=[0],
        accelerator="gpu",
        strategy="auto",
        max_epochs=config.get_epochs(),
        logger=tensorboard_logger,
        callbacks=callbacks,
        log_every_n_steps=14,
        deterministic=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(net)

    print(f"Train completed, best_metric: {net.best_val_dice:.4f} at epoch {net.best_val_epoch}")

    trainer.test(ckpt_path="best")


if __name__ == '__main__':
    main()

import logging
import argparse
import os
import pandas as pd

from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from monai.inferers import sliding_window_inference

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
from utils.helpers import load_images_from_path, check_dataset
from config.constants import (COVID_CASES_PATH, INFECTION_MASKS_PATH, SEED, VALIDATION_INFERENCE_ROI_SIZE, SPATIAL_SIZE,
                              LUNG_MASKS_PATH, EXPERIMENTS_PATH, COVID_PREPROCESSED_CASES_PATH,
                              INFECTION_PREPROCESSED_MASKS_PATH)
import torch
import numpy as np
from monai.metrics import DiceMetric
import lightning as L

torch.set_float32_matmul_precision('high')


class Net(L.pytorch.LightningModule):
    def __init__(self, learning_rate: float, model: torch.nn.Module, loss_function: torch.nn, volumes_path: str,
                 masks_path: str, experiment_name: str):
        super(Net, self).__init__()
        print(f"Using lr: {learning_rate}")
        self.learning_rate = learning_rate
        self.experiment_name = experiment_name

        self.save_hyperparameters(ignore=["model", "loss_function", "volumes_path", "masks_path", "experiment_name"])

        self.post_pred = monai.transforms.Compose(
            [monai.transforms.EnsureType(data_type='tensor'), monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)])
        self.post_label = monai.transforms.Compose([monai.transforms.AsDiscrete(threshold=0.5)])

        self.best_val_dice = 0
        self.best_val_epoch = 0

        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.test_step_outputs = []
        self.test_dump_data_frame = []

        self.test_paths = None
        self.val_paths = None
        self.train_paths = None

        self.training_ds = None
        self.validation_ds = None
        self.test_ds = None

        self.volumes_path = volumes_path
        self.masks_path = masks_path

        self.model = model
        print(f"Using model: {type(self.model)}")
        self.loss_function = loss_function

        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.train_dice_metric = DiceMetric(include_background=True, reduction="mean")

    def forward(self, x):
        return self.model(x)

    def prepare_data(self) -> None:
        # Load images and masks
        logging.info(f"Loading images from {self.volumes_path} and masks from {self.masks_path}")
        images = load_images_from_path(self.volumes_path)
        labels = load_images_from_path(self.masks_path)

        # Convert images and masks to a list of dictionaries with keys "img" and "mask"
        data_dicts = np.array([{"img": img, "mask": mask} for img, mask in zip(images, labels)])
        logging.debug(data_dicts)

        shuffler = np.random.RandomState(SEED)
        shuffler.shuffle(data_dicts)
        data_dicts = list(data_dicts)

        # Split the data into training (70%), validation (20%), and test sets (10%)
        test_split = int(len(data_dicts) * 0.1)
        val_split = int(len(data_dicts) * 0.2)

        self.train_paths = data_dicts[test_split + val_split:]
        self.val_paths = data_dicts[test_split:test_split + val_split]
        self.test_paths = data_dicts[:test_split]

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
        raw_outputs = self.forward(inputs)
        loss = self.loss_function(raw_outputs, labels)
        self.log("train_step_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        outputs = [self.post_pred(i) for i in decollate_batch(raw_outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.train_dice_metric(y_pred=outputs, y=labels)

        self.log("train_step_dice", self.train_dice_metric.aggregate().item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)

        train_loss_dictionary = {"loss": loss}
        self.train_step_outputs.append(train_loss_dictionary)
        return train_loss_dictionary

    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack([i["loss"] for i in self.train_step_outputs]).mean()
        mean_train_dice = self.train_dice_metric.aggregate().item()
        self.train_dice_metric.reset()

        self.log_dict({"train_dice": mean_train_dice, "train_loss": avg_loss}, on_epoch=True, on_step=False, prog_bar=True)

        # logs = {
        #     "train_dice": mean_train_dice,
        #     "train_loss": avg_loss,
        # }

        self.logger.experiment.add_scalars("losses", {"train": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("dice", {"train": mean_train_dice}, self.current_epoch)
        # self.loggers[0].log_metrics(logs, step=self.current_epoch)

        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["mask"]
        roi_size = VALIDATION_INFERENCE_ROI_SIZE
        sw_batch_size = 4

        outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, self.forward, overlap=0.3)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)

        validation_loss_dictionary = {"val_loss": loss}
        self.validation_step_outputs.append(validation_loss_dictionary)
        return validation_loss_dictionary

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        self.log_dict({"val_dice": mean_val_dice, "val_loss": avg_loss}, prog_bar=True, on_epoch=True, on_step=False)

        # logs = {
        #     "val_dice": mean_val_dice,
        #     "val_loss": avg_loss,
        # }

        self.logger.experiment.add_scalars("losses", {"val_loss": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("dice", {"val_dice": mean_val_dice}, self.current_epoch)
        # self.loggers[0].log_metrics(logs, step=self.current_epoch)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["mask"]
        roi_size = VALIDATION_INFERENCE_ROI_SIZE
        sw_batch_size = 4

        outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, self.forward, overlap=0.6)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        volume_dice = self.dice_metric.aggregate().item()

        # Create a pandas dataframe with batch_idx, test_loss, and test_metric columns
        df = pd.DataFrame({"volume": [batch_idx], "test_loss": [loss.item()], "test_metric": [volume_dice]})
        self.test_dump_data_frame.append(df)
        self.test_step_outputs.append({"test_loss": loss})

    def on_test_epoch_end(self):
        test_loss = torch.stack([x["test_loss"] for x in self.test_step_outputs]).mean()
        test_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        self.log_dict({"test_dice": test_dice, "test_loss": test_loss}, prog_bar=True, logger=False)

        df = pd.concat(self.test_dump_data_frame)
        df.to_csv(EXPERIMENTS_PATH + self.experiment_name + "/test_metrics.csv", header=True, index=False)


def main():
    # Initialize the configuration
    config = Config()
    config.set_logging_verbose()

    # Set the seed for reproducibility
    L.seed_everything(SEED)
    torch.manual_seed(SEED)
    monai.utils.set_determinism(seed=SEED)

    # Experiment name
    experiment_name = f"{config.get_network_name()}_{config.get_loss_name()}_2"
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
    # train_csv_logger = L.pytorch.loggers.CSVLogger(save_dir=EXPERIMENTS_PATH, name=experiment_name, version="csv_train")

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

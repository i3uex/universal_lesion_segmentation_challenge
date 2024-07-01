import logging
import argparse
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from monai.inferers import sliding_window_inference

import nets.nets
from monai.metrics.meandice import compute_dice
from config.config import Config
from preprocessing.covid_dataset import CovidDataset
import monai.data
from monai.data import DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from preprocessing.transforms import get_hrct_transforms, get_cbct_transforms, \
    get_val_hrct_transforms, get_val_cbct_transforms
from utils.helpers import load_images_from_path, check_dataset
from config.constants import (COVID_CASES_PATH, INFECTION_MASKS_PATH, SEED, VALIDATION_INFERENCE_ROI_SIZE, SPATIAL_SIZE,
                              LUNG_MASKS_PATH)
import torch
import numpy as np
from monai.metrics import DiceMetric
import lightning as L

torch.set_float32_matmul_precision('high')


class Net(L.pytorch.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.config = None

        self.post_pred = monai.transforms.Compose(
            [monai.transforms.EnsureType(data_type='tensor'), monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)])
        self.post_label = monai.transforms.Compose([monai.transforms.AsDiscrete(threshold=0.5)])

        self.best_val_dice = 0
        self.best_val_epoch = 0

        self.validation_step_outputs = []
        self.train_step_outputs = []

        self.training_ds = None
        self.validation_ds = None
        self.test_ds = None

        self.model = None
        self.dice_metric = None
        self.train_dice_metric = None
        self.loss_function = None

    def configure_training_process(self):
        self.model = self.config.get_network()
        logging.info(f"Using model: {self.model}")

        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.train_dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.loss_function = monai.losses.DiceLoss(sigmoid=True)

    def forward(self, x):
        return self.model(x)

    def prepare_data(self) -> None:
        # Load images and masks
        logging.info(f"Loading images from {COVID_CASES_PATH}")
        images = load_images_from_path(COVID_CASES_PATH)
        labels = load_images_from_path(self.config.get_mask_data_path())

        # Convert images and masks to a list of dictionaries with keys "img" and "mask"
        data_dicts = np.array([{"img": img, "mask": mask} for img, mask in zip(images, labels)])
        logging.debug(data_dicts)

        shuffler = np.random.RandomState(SEED)
        shuffler.shuffle(data_dicts)
        data_dicts = list(data_dicts)

        # Split the data into training (70%), validation (20%), and test sets (10%)
        test_split = int(len(data_dicts) * 0.1)
        val_split = int(len(data_dicts) * 0.2)

        train_paths = data_dicts[test_split + val_split:]
        val_paths = data_dicts[test_split:test_split + val_split]
        test_paths = data_dicts[:test_split]

        # Define the CovidDataset instances for training, validation, and test
        self.training_ds = CovidDataset(volumes=train_paths, hrct_transform=get_hrct_transforms(),
                                        cbct_transform=get_cbct_transforms())
        self.validation_ds = CovidDataset(volumes=val_paths, hrct_transform=get_val_hrct_transforms(),
                                          cbct_transform=get_val_cbct_transforms())
        self.test_ds = CovidDataset(volumes=test_paths, hrct_transform=get_val_hrct_transforms(),
                                    cbct_transform=get_val_cbct_transforms())

        # Check the dataset
        print("Checking the dataset")
        check_dataset(self.validation_ds)
        check_dataset(self.test_ds)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.training_ds, batch_size=1, num_workers=4)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.validation_ds, batch_size=1, num_workers=4)
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_ds, batch_size=1, num_workers=4)
        return test_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["img"].cuda(), batch["mask"].cuda()
        raw_outputs = self.forward(inputs)
        loss = self.loss_function(raw_outputs, labels)
        self.log("train_step_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        outputs = [self.post_pred(i) for i in decollate_batch(raw_outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.train_dice_metric(y_pred=outputs, y=labels)

        self.log("train_step_dice", self.train_dice_metric.aggregate().item(), on_step=True, on_epoch=False, prog_bar=True)

        train_loss_dictionary = {"loss": loss}
        self.train_step_outputs.append(train_loss_dictionary)
        return train_loss_dictionary

    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack([i["loss"] for i in self.train_step_outputs]).mean()
        mean_train_dice = self.train_dice_metric.aggregate().item()
        self.train_dice_metric.reset()

        self.log_dict({"train_dice": mean_train_dice, "train_loss": avg_loss}, prog_bar=True)

        tensorboard_logs = {
            "train_dice": mean_train_dice,
            "train_loss": avg_loss,
        }

        self.logger.experiment.add_scalars("losses", {"train": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("dice", {"train": mean_train_dice}, self.current_epoch)
        self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)

        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["mask"]
        roi_size = VALIDATION_INFERENCE_ROI_SIZE
        sw_batch_size = 4

        outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)

        self.log("val_step_dice", self.dice_metric.aggregate().item(), on_step=True, on_epoch=False, prog_bar=True)

        validation_loss_dictionary = {"val_loss": loss}
        self.validation_step_outputs.append(validation_loss_dictionary)
        return validation_loss_dictionary

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        self.log_dict({"val_dice": mean_val_dice, "val_loss": avg_loss}, prog_bar=True, on_epoch=True, on_step=False)

        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": avg_loss,
        }

        self.logger.experiment.add_scalars("losses", {"val_loss": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("dice", {"val_dice": mean_val_dice}, self.current_epoch)
        self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
            inputs, labels = batch["img"], batch["mask"]
            roi_size = VALIDATION_INFERENCE_ROI_SIZE
            sw_batch_size = 4

            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, self.forward)
            loss = self.loss_function(outputs, labels)
            outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
            labels = [self.post_label(i) for i in decollate_batch(labels)]
            self.dice_metric(y_pred=outputs, y=labels)

            return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        self.log_dict({"test_dice": test_dice, "test_loss": test_loss}, prog_bar=True)

        tensorboard_logs = {
            "test_dice": test_dice,
            "test_loss": test_loss,
        }

        self.logger.experiment.add_scalars("losses", {"test_loss": test_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("dice", {"test_dice": test_dice}, self.current_epoch)
        self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)


def main():
    # Initialize the configuration
    config = Config()
    config.set_logging_verbose()

    # Set the seed for reproducibility
    L.seed_everything(SEED)

    net = Net()
    net.config = config
    net.configure_training_process()

    tensorboard_logger = (L.pytorch.loggers.TensorBoardLogger(save_dir="lightning_logs", name="lightning_logs"))
    callbacks = [L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")]

    trainer = L.pytorch.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=config.get_epochs(),
        logger=tensorboard_logger,
        callbacks=callbacks,
        log_every_n_steps=14,
        deterministic=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(net, ckpt_path=config.get_checkpoint())

    print(f"Train completed, best_metric: {net.best_val_dice:.4f} at epoch {net.best_val_epoch}")

    trainer.test(dataloaders=net.test_dataloader(), ckpt_path="best")


if __name__ == '__main__':
    main()

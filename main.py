import logging
import argparse
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, STEP_OUTPUT
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from preprocessing.covid_dataset import CovidDataset
import monai.data
from monai.data import DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from preprocessing.transforms import get_hrct_transforms, get_cbct_transforms, \
    get_val_hrct_transforms, get_val_cbct_transforms
from utils.helpers import load_images_from_path, check_dataset
from config.constants import (COVID_CASES_PATH, INFECTION_MASKS_PATH, SEED)
from config.config import Config
import torch
import numpy as np
from monai.metrics import DiceMetric
import lightning as L

class Net(L.pytorch.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.train_dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.loss_function = monai.losses.DiceLoss(sigmoid=True)
        self.post_pred = monai.transforms.Compose(
            [monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold_values=0.5)])
        self.post_label = monai.transforms.Compose([monai.transforms.AsDiscrete(threshold_values=0.5)])
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.training_ds = None
        self.validation_ds = None
        self.test_ds = None

    def forward(self, x):
        return self.model(x)

    def prepare_data(self) -> None:
        # Load images and masks
        logging.info(f"Loading images from {COVID_CASES_PATH}")
        images = load_images_from_path(COVID_CASES_PATH)
        labels = load_images_from_path(INFECTION_MASKS_PATH)

        # Convert images and masks to a list of dictionaries with keys "img" and "mask"
        data_dicts = np.array([{"img": img, "mask": mask} for img, mask in zip(images, labels)])
        logging.debug(data_dicts)

        shuffler = np.random.RandomState(SEED)
        shuffler.shuffle(data_dicts)
        data_dicts = list(data_dicts)

        # Split the data into training (70%), validation (10%), and test sets (20%)
        test_split = int(len(data_dicts) * 0.2)
        val_split = int(len(data_dicts) * 0.1)

        train_paths = data_dicts[test_split + val_split:]
        val_paths = data_dicts[test_split:test_split + val_split]
        test_paths = data_dicts[:test_split]
        test_paths = data_dicts[:test_split]

        # Check the dataset
        print("Checking the dataset")
        check_dataset(train_paths)
        check_dataset(val_paths)
        check_dataset(test_paths)
        print("Dataset is valid")

        # Define the CovidDataset instances for training, validation, and test
        self.training_ds = CovidDataset(volumes=train_paths, hrct_transform=get_hrct_transforms(),
                                        cbct_transform=get_cbct_transforms())
        self.validation_ds = CovidDataset(volumes=val_paths, hrct_transform=get_val_hrct_transforms(),
                                          cbct_transform=get_val_cbct_transforms())
        self.test_ds = CovidDataset(volumes=test_paths, hrct_transform=get_val_hrct_transforms(),
                                    cbct_transform=get_val_cbct_transforms())

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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["img"].to(self.device), batch["mask"].to(self.device)
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, labels)
        #outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        # labels = [self.post_label(i) for i in decollate_batch(labels)]
        # dice = self.train_dice_metric(y_pred=outputs, y=labels)
        # train_dictionary = {"loss": loss, "number": len(outputs)}
        # self.train_step_outputs.append(train_dictionary)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    # def on_training_epoch_end(self):
    #     train_loss, num_items = 0, 0
    #     for output in self.train_step_outputs:
    #         train_loss += output["loss"].sum().item()
    #         num_items += output["number"]
    #
    #     mean_train_loss = torch.tensor(train_loss / num_items)
    #     self.log("train_loss", mean_train_loss, prog_bar=True)
    #     train_dice = self.train_dice_metric.aggregate().item()
    #     self.train_dice_metric.reset()
    #     self.log("train_dice", train_dice, prog_bar=True)
    #
    #     tensorboard_logs = {
    #         "train_dice": train_dice,
    #         "loss": mean_train_loss,
    #     }
    #
    #     self.train_step_outputs.clear()
    #     return {"log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["mask"]
        roi_size = (64, 64, 64)
        sw_batch_size = 4

        outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        # labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)

        validation_loss_dictionary = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(validation_loss_dictionary)
        return validation_loss_dictionary

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_dice", mean_val_dice, prog_bar=True)

        mean_val_loss = torch.tensor(val_loss / num_items)
        self.log("val_loss", val_loss / num_items, prog_bar=True)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        # print(
        #     f"current epoch: {self.current_epoch} "
        #     f"current mean dice: {mean_val_dice:.4f}"
        #     f"\nbest mean dice: {self.best_val_dice:.4f} "
        #     f"at epoch: {self.best_val_epoch}"
        # )

        self.validation_step_outputs.clear()
        return {"log": tensorboard_logs}


def main():
    parser = argparse.ArgumentParser(description="COVID-19 Segmentation")
    parser.add_argument('--architecture', type=str, default='unet', help='Model arch: unet, resnet, etc.')
    parser.add_argument('--metrics', type=str, default='dice', help='Metrics to use: dice, iou, etc.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--mask_data_path', type=str, default=INFECTION_MASKS_PATH, help='Path to the dataset')
    parser.add_argument('--verbose', type=str, help='Select output verbosity [INFO, DEBUG, ERROR, WARNING]')
    config = parser.parse_args()

    L.seed_everything(SEED)

    net = Net()

    tensorboard_logger = (L.pytorch.loggers.TensorBoardLogger(save_dir="lightning_logs", name="lightning_logs"))
    callbacks = [L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")]

    trainer = L.pytorch.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=config.epochs,
        logger=tensorboard_logger,
        callbacks=callbacks,
        log_every_n_steps=14,
        deterministic=True,
    )

    trainer.fit(net)

    print(f"Train completed, best_metric: {net.best_val_dice:.4f} " f"at epoch {net.best_val_epoch}")


if __name__ == '__main__':
    main()
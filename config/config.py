import logging
import sys

from config.constants import ZENODO_INFECTION_MASKS_PATH, SEED
from experiments_items.nets import *
from experiments_items.loss_functions import *
import argparse

HELP_MESSAGE = '''
COVID-19 Segmentation
---------------------
This script is used to train a model to segment COVID-19 infections in CT scans.
The script uses a U-Net architecture and the Dice loss function by default.

Arguments:
    --architecture: Model arch: unet, unetr, swin_unet.
    --loss: Loss function to use: dice, crossentropy, hausdorff, generalizeddicefocal.
    --epochs: Number of epochs to train
    --batch_size: Batch size for training
    --learning_rate: Learning rate for optimizer
    --mask_data_path: Path to the dataset
    --verbose: Select output verbosity [INFO, DEBUG, ERROR, WARNING]
    --seed: Seed for reproducibility
    --help: Show this help message and exit
'''

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="COVID-19 Segmentation")
        self.parser.add_argument('--architecture', type=str, default='unet', help='Model arch: unet, resnet, etc.')
        self.parser.add_argument('--loss', type=str, default='dice', help='Loss function to use: dice, crossentropy, etc.')
        self.parser.add_argument('--epochs', type=int, default=1300, help='Number of epochs to train')
        self.parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
        self.parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
        self.parser.add_argument('--mask_data_path', type=str, default=ZENODO_INFECTION_MASKS_PATH, help='Path to the dataset')
        self.parser.add_argument('--verbose', type=str, help='Select output verbosity [INFO, DEBUG, ERROR, WARNING]')
        self.parser.add_argument('--seed', type=int, default=SEED, help='Seed for reproducibility')
        self.args = self.parser.parse_args()

    def get_args(self):
        return self.args

    def get_learning_rate(self):
        return self.args.learning_rate

    def get_epochs(self):
        return self.args.epochs

    def set_logging_verbose(self):
        log = {
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
        }
        # logging.basicConfig(stream=sys.stdout, level=log[self.args.verbose.upper()], handlers=[logging.StreamHandler()])

    def get_network_name(self):
        return self.args.architecture

    def get_network(self):
        if self.args.architecture == 'unet':
            return covid_unet
        elif self.args.architecture == 'unetr':
            return covid_unetr
        elif self.args.architecture == 'swin_unet':
            return covid_swin_unet

    def get_loss_name(self):
        return self.args.loss

    def get_loss(self):
        if self.args.loss == 'crossentropy':
            return cross_entropy_loss
        elif self.args.loss == 'dice':
            return dice_loss
        elif self.args.loss == 'hausdorff':
            return hausdorf_loss
        elif self.args.loss == 'generalizeddicefocal':
            return generalize_dice_focal_loss

    def get_mask_data_path(self):
        return self.args.mask_data_path

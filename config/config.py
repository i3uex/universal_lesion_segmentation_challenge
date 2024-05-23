import argparse
import logging
from config.constants import INFECTION_MASKS_PATH

#TODO: Make use of the Config class to configure the training process
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="COVID-19 Segmentation")
        self.parser.add_argument('--architecture', type=str, default='unet', help='Model arch: unet, resnet, etc.')
        self.parser.add_argument('--metrics', type=str, default='dice', help='Metrics to use: dice, iou, etc.')
        self.parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
        self.parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
        self.parser.add_argument('--mask_data_path', type=str, default=INFECTION_MASKS_PATH, help='Path to the dataset')
        self.parser.add_argument('--verbose', type=str, help='Select output verbosity [INFO, DEBUG, ERROR, WARNING]')
        self.args = self.parser.parse_args()

    def get_args(self):
        return self.args

    def set_logging_verbose(self):
        log = {
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
        }
        logging.basicConfig(level=log[self.args.verbose.upper()], handlers=[logging.StreamHandler()])

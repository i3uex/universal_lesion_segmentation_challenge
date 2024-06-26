import logging
import sys
from nets import nets


#TODO: Make use of the Config class to configure the training process
class Config:
    def __init__(self, args=None):
        self.args = args

    def get_args(self):
        return self.args

    def get_epochs(self):
        return self.args.epochs

    def set_logging_verbose(self):
        log = {
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
        }
        logging.basicConfig(stream=sys.stdout, level=log[self.args.verbose.upper()], handlers=[logging.StreamHandler()])

    def get_network(self):
        if self.args.architecture == 'unet':
            return nets.covid_unet
        elif self.args.architecture == 'unetr':
            return nets.covid_unetr
import logging
import sys
#TODO: Make use of the Config class to configure the training process
class Config:
    def __init__(self, args=None):
        self.args = args

    def get_args(self):
        return self.args

    def set_logging_verbose(self):
        log = {
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
        }
        logging.basicConfig(stream=sys.stdout, level=log[self.args.verbose.upper()], handlers=[logging.StreamHandler()])

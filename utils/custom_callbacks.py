from lightning.pytorch.callbacks import Callback
import time


class CustomTimingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        print("Training is starting...")

    def on_train_end(self, trainer, pl_module):
        self.end = time.time()
        days, hours, minutes, seconds = self.seconds_to_readable_time(self.end - self.start_time)
        print(f"Training is done! Time elapsed: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")

    def seconds_to_readable_time(self, seconds: float) -> tuple:
        """
        Convert a given time duration from seconds to a more readable format.

        This method takes an input time duration in seconds and converts it into days, hours, minutes, and seconds.
        It uses floor division and the modulus operator to calculate the equivalent time in days, hours, minutes,
        and seconds.

        Args:
            seconds (int): The time duration in seconds.

        Returns:
            tuple: A tuple containing the time duration in days, hours, minutes, and seconds.
        """
        days = seconds // (24 * 3600)
        seconds %= (24 * 3600)
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return days, hours, minutes, seconds

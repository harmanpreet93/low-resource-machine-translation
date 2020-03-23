import typing
import datetime
from model_logging import get_logger
import tensorflow as tf


class MainModel(tf.keras.Model):
    TRAINING_REQUIRED = False

    def __init__(
            self,
            stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
            target_time_offsets: typing.List[datetime.timedelta],
            config: typing.Dict[typing.AnyStr, typing.Any],
            return_ghi_only=True
    ):
        """
        Args:
            stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation)
            target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
            config: configuration dictionary holding any extra parameters that might be required by the user. These
                parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
                such a JSON file is completely optional, and this argument can be ignored if not needed.
        """
        super(MainModel, self).__init__()
        self.stations = stations
        self.target_time_offsets = target_time_offsets
        self.config = config
        self.initialize()

    def initialize(self):
        self.logger = get_logger()
        self.logger.debug("Model start")

    def call(self, inputs):
        '''
        Defines the forward pass through our model
        '''
        # image = inputs[0]
        clearsky_GHIs = inputs[1]
        # true_GHIs = inputs[2]
        return clearsky_GHIs

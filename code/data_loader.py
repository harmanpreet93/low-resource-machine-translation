import h5py
import typing
import datetime
import numpy as np
import tensorflow as tf
from numpy.core._multiarray_umath import ndarray
from model_logging import get_logger
import glob


class DataLoader():

    def __init__(
            self,
            dataframe: pd.DataFrame,
            target_datetimes: typing.List[datetime.datetime],
            stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
            target_time_offsets: typing.List[datetime.timedelta],
            config: typing.Dict[typing.AnyStr, typing.Any],
            data_folder: typing.AnyStr
    ):
        """
        Copy-paste from evaluator.py:
        Args:
            dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset) for all
                relevant timestamp values over the test period.
            target_datetimes: a list of timestamps that your data loader should use to provide imagery for your model.
                The ordering of this list is important, as each element corresponds to a sequence of GHI values
                to predict. By definition, the GHI values must be provided for the offsets given by
                ``target_time_offsets`` which are added to each timestamp (T=0) in this datetimes list.
            stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation)
            target_time_offsets: the list of time-deltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
            config: configuration dictionary holding extra parameters
        """
        self.config = config
        self.data_folder = data_folder
        self.initialize()

    def initialize(self):
        self.logger = get_logger()
        self.logger.debug("Initialize start")
        self.test_station = self.stations[0]

        self.data_loader = tf.data.Dataset.from_generator(
            self.data_generator_fn,
            output_types=(tf.int32, tf.int32)
        ).prefetch(tf.data.experimental.AUTOTUNE)

    def data_generator_fn(self):
        for f_path in self.data_files_list:
            with h5py.File(f_path, 'r') as h5_data:
                yield "TODO"

    def get_data_loader(self):
        '''
        Returns:
            A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
            must correspond to one sequence of past imagery data. The tensors must be generated in the order given
            by ``target_sequences``.
        '''
        return self.data_loader

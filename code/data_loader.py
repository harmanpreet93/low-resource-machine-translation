import h5py
import typing
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
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
        self.dataframe = dataframe
        self.target_datetimes = target_datetimes
        self.stations = list(stations.keys())
        self.config = config
        self.target_time_offsets = target_time_offsets
        self.data_folder = data_folder
        self.initialize()

    def initialize(self):
        self.logger = get_logger()
        self.logger.debug("Initialize start")
        self.test_station = self.stations[0]
        self.output_seq_len = len(self.target_time_offsets)
        self.data_files_list = glob.glob(self.data_folder + "/*.hdf5")
        # sort required for evaluator script
        self.data_files_list.sort()

        stations = np.array([b"BND", b"TBL", b"DRA", b"FPK", b"GWN", b"PSU", b"SXF"])
        self.encoder = OneHotEncoder(sparse=False)
        stations = stations.reshape(len(stations), 1)
        self.encoder.fit(stations)

        self.data_loader = tf.data.Dataset.from_generator(
            self.data_generator_fn,
            output_types=(tf.float32, tf.float32, tf.float32, tf.bool, tf.float32, tf.float32, tf.float32)
        ).prefetch(tf.data.experimental.AUTOTUNE)

    def to_cyclical_secondofday(self, date):
        SECONDS_PER_DAY = 24 * 60 * 60
        second_of_day = (date.hour * 60 + date.minute) * 60 + date.second
        day_cycle_rad = second_of_day / SECONDS_PER_DAY * 2.0 * np.pi
        day_cycle_x = np.sin(day_cycle_rad)
        day_cycle_y = np.cos(day_cycle_rad)
        return pd.DataFrame(day_cycle_x), pd.DataFrame(day_cycle_y)

    def to_cyclical_dayofyear(self, date):
        DAYS_PER_YEAR = 365
        year_cycle_rad = date.dayofyear / DAYS_PER_YEAR
        year_cycle_x = np.sin(year_cycle_rad)
        year_cycle_y = np.cos(year_cycle_rad)
        return pd.DataFrame(year_cycle_x), pd.DataFrame(year_cycle_y)

    def create_sin_cos(self, date):
        date = date.astype('U50')
        date = pd.to_datetime(date.flatten())
        day_cycle_x, day_cycle_y = self.to_cyclical_secondofday(date)
        year_cycle_x, year_cycle_y = self.to_cyclical_dayofyear(date)
        return np.array(pd.concat((day_cycle_x, day_cycle_y, year_cycle_x, year_cycle_y), axis=1))

    def get_onehot_station_id(self, station_ids):
        return self.encoder.transform(station_ids)

    def data_generator_fn(self):
        for f_path in self.data_files_list:
            # f_path = os.path.join(self.data_folder, file)

            with h5py.File(f_path, 'r') as h5_data:
                images = np.array(h5_data["images"])
                true_GHIs: ndarray = np.array(h5_data["GHI"])
                clearsky_GHIs = np.array(h5_data["clearsky_GHI"])
                station_ids = np.array(h5_data["station_id"])
                night_flags = np.array(h5_data["night_flags"]).astype(np.bool)
                station_id_onehot = self.get_onehot_station_id(station_ids)
                date = np.array(h5_data['datetime_sequence'])
                date_vector = self.create_sin_cos(date)  # size: batch * 4

                yield images, clearsky_GHIs, true_GHIs, night_flags, station_id_onehot, date_vector, true_GHIs

    def get_data_loader(self):
        '''
        Returns:
            A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
            must correspond to one sequence of past imagery data. The tensors must be generated in the order given
            by ``target_sequences``.
        '''
        return self.data_loader

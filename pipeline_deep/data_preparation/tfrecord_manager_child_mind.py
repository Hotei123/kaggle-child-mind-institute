import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
from pipeline_deep.data_preparation.tfrecord_manager import TFRecordManager


class TFRecordManagerChildMind(TFRecordManager):

    def __init__(self, config):
        # TODO: avoid overriding the constructor
        
        # Train data
        self.data_non_temp_train = pd.read_csv('output/train_processed.csv')
        self.ids_train = pd.read_csv(config['prepare_data']['path_tabular_train'])['id']
        # Filling NaN labels
        path_labels_train_filled = pathlib.Path(config['prepare_data']['path_output']).joinpath('labels_filled.csv')
        labels_train_filled: pd.DataFrame = pd.read_csv(path_labels_train_filled)
        self.data_non_temp_train.sii = labels_train_filled.sii
        # Submit data
        self.data_non_temp_submit = pd.read_csv('output/test_processed.csv')
        self.ids_submit = pd.read_csv(config['prepare_data']['path_tabular_test'])['id']
        for col in self.data_non_temp_train.columns:
            if col not in self.data_non_temp_submit.columns:
                self.data_non_temp_submit[col] = 0

        for col in self.data_non_temp_submit.columns:
            if col not in self.data_non_temp_train.columns:
                self.data_non_temp_train[col] = 0

        self.path_non_temporal_submit = config['prepare_data']['path_tabular_test']
        
        self.n_examples_train: int = self.data_non_temp_train.shape[0]
        self.n_examples_per_file_train: int = config['prepare_data']['n_examples_per_file_train']
        self.n_examples_submit: int = self.data_non_temp_submit.shape[0]
        self.n_examples_per_file_submit: int = config['prepare_data']['n_examples_per_file_submit']
        self.n_cols_ts: int = config['prepare_data']['n_cols_ts']
        self.n_rows_ts: int = config['prepare_data']['n_rows_ts']

        self.path_output: str = pathlib.Path(config['prepare_data']['path_output']).joinpath('tfrecords')
        self.vars_dummy = pd.read_csv('output/dummy_var_names.csv').dummy_names.to_list()
        self.vars_num = config['prepare_data']['vars_num']
        self.vars_cat = config['prepare_data']['vars_cat']
        self.var_target = config['prepare_data']['var_target']
        self.vars_time_desc = [x for x in self.data_non_temp_train.columns if x.startswith('series_desc_')]
        self.vars_num_cat_time_desc = config['prepare_data']['vars_num'] + self.vars_dummy + self.vars_time_desc
        
        self.feature_description = {var_name: tf.io.FixedLenFeature([], tf.float32) for var_name in self.vars_num_cat_time_desc}
        self.feature_description['ts'] = tf.io.FixedLenFeature([self.n_rows_ts * self.n_cols_ts], tf.float32)
        self.feature_description['sii'] = tf.io.FixedLenFeature([], tf.float32)
        self.path_series_train = config['prepare_data']['path_series_train']
        self.path_series_test = config['prepare_data']['path_series_test']

    def get_example(self, index: int, prefix: str) -> tf.train.Example:
        # This function returns an example from the raw data.
        if prefix == 'train':
            example = self.data_non_temp_train.iloc[index].to_dict()
        elif prefix == 'submit':
            example = self.data_non_temp_submit.iloc[index].to_dict()
        # TODO: use all the tabular and time series variables for writing the TFRecords
        # TODO: write categorical variables
        # TODO: normalize data previous to writing the TFRecords
        feature = {var_name: self._float_feature(example[var_name]) if not np.isnan(example[var_name]) else self._float_feature(0) 
                   for var_name in self.vars_num_cat_time_desc}
        if prefix == 'train':
            path_series = pathlib.Path(self.path_series_train).joinpath(f'id={self.ids_train.iloc[index]}/part-0.parquet')
        else:
            path_series = pathlib.Path(self.path_series_test).joinpath(f'id={self.ids_submit.iloc[index]}/part-0.parquet')
        if path_series.exists():
            data_series_row = pd.read_parquet(path_series)
            data_series_row.drop(columns='step', inplace=True)
            data_series_row = data_series_row.iloc[:self.n_rows_ts, :]
            if data_series_row.shape[0] < self.n_rows_ts:
                pad_zeros = pd.DataFrame(np.zeros((self.n_rows_ts - data_series_row.shape[0], self.n_cols_ts)), 
                                         columns=data_series_row.columns)
                data_series_row = pd.concat([data_series_row, pad_zeros])
                data_series_row.fillna(0, inplace=True)
            feature['ts'] = self._float_feature_list(data_series_row.values.flatten())
            # feature['ts'] = self._float_feature_list(np.zeros(self.n_rows_ts * self.n_cols_ts))
        else:
            feature['ts'] = self._float_feature_list(np.zeros(self.n_rows_ts * self.n_cols_ts))
        if 'sii' in example:
            if np.isnan(example['sii']):
                feature['sii'] = self._float_feature(0)
            else:
                feature['sii'] = self._float_feature(example['sii'])
        else:
            feature['sii'] = self._float_feature(0)
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def parse_example(self, example: tf.train.Example):
        # Returns the parsed data from the input `tf.train.Example` proto.
        # TODO: normalize training
        # TODO: add time series data (first from describe.(), and then the whole series)
        example_parsed = tf.io.parse_single_example(example, self.feature_description)
        x_0 = tf.stack([example_parsed[var] for var in self.vars_num], axis=0)
        x_1 = tf.stack([example_parsed[var] for var in self.vars_dummy], axis=0)
        x_2 = tf.stack([example_parsed[var] for var in self.vars_time_desc], axis=0)
        # print(f'\n\n\nType: {type(example_parsed["ts"])}. Shape: {example_parsed["ts"].shape}. {"ts" in example_parsed}\n\n\n')
        x_3 = tf.reshape(example_parsed['ts'], (self.n_rows_ts, self.n_cols_ts))
        return (x_0, x_1, x_2, x_3), example_parsed['sii']
        # return (x_0, x_1, x_2), example_parsed['sii']
    
    @staticmethod
    def normalization_function(dataset):
        # TODO: normalize only training, and use parameters in validation or test. Since the shallow data
        #  was already normalized, only the temporal data needs to be normalized.

        x_0_max = None
        x_2_max = None
        x_0_min = None
        x_2_min = None

        for (x_0, x_1, x_2, x_3), y in dataset:
            if x_0_max is None:
                x_0_max = x_0
                x_2_max = x_2
                x_0_min = x_0
                x_2_min = x_2
            else:
                x_0_max = np.max([x_0_max, x_0], axis=0)
                x_2_max = np.max([x_2_max, x_2], axis=0)
                x_0_min = np.min([x_0_min, x_0], axis=0)
                x_2_min = np.min([x_2_min, x_2], axis=0)

        diff_0 = x_0_max - x_0_min
        diff_0[diff_0 == 0] = 1
        diff_2 = x_2_max - x_2_min
        diff_2[diff_2 == 0] = 1
        return lambda x, y: (((x[0] - x_0_min) / diff_0, x[1], (x[2] - x_2_min) / diff_2, x[3]), y)

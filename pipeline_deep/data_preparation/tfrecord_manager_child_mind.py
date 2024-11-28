import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
from pipeline_deep.data_preparation.tfrecord_manager import TFRecordManager


class TFRecordManagerChildMind(TFRecordManager):

    def __init__(self, config):
        # TODO: avoid overriding the constructor
        path_non_temporal_train = config['prepare_data']['path_tabular_train']
        self.path_non_temporal_submit = config['prepare_data']['path_tabular_test']
        # Train data
        self.data_non_temp_train: pd.DataFrame = pd.read_csv(path_non_temporal_train)
        self.n_examples_train: int = self.data_non_temp_train.shape[0]
        self.n_examples_per_file_train: int = config['prepare_data']['n_examples_per_file_train']
        # Submit data
        self.data_non_temp_submit: pd.DataFrame = pd.read_csv(self.path_non_temporal_submit)
        self.n_examples_submit: int = self.data_non_temp_submit.shape[0]
        self.n_examples_per_file_submit: int = config['prepare_data']['n_examples_per_file_submit']

        self.path_output: str = pathlib.Path(config['prepare_data']['path_output']).joinpath('tfrecords')
        self.feature_description = {var_name: tf.io.FixedLenFeature([], tf.float32) for var_name in config['prepare_data']['vars_num']}
        self.feature_description['sii'] = tf.io.FixedLenFeature([], tf.float32)
        self.vars_num = config['prepare_data']['vars_num']
        self.var_target = config['prepare_data']['var_target']

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
                   for var_name in self.vars_num}
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
        # TODO: make predictions of missing target vars
        example_parsed = tf.io.parse_single_example(example, self.feature_description)
        x1 = tf.stack([example_parsed[var] for var in self.vars_num], axis=0)
        x2 = tf.stack([example_parsed['Physical-BMI'], 
                       example_parsed['Physical-Height'],
                       example_parsed['Physical-Weight']], axis=0)
        x1 = tf.where(tf.math.is_nan(x1), 0.0, x1)
        x2 = tf.where(tf.math.is_nan(x2), 0.0, x2)
        return (x1, x2), example_parsed['sii']

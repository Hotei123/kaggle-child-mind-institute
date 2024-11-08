import os
import shutil
import pandas as pd
from data_preparation.tfrecord_manager import TFRecordManager
import tensorflow as tf
import pathlib


class TFRecordManagerChildMind(TFRecordManager):

    def __init__(self, config):
        # TODO: avoid overriding the constructor
        path_non_temporal_train = config['prepare_data']['path_tabular_train']
        path_non_temporal_submit = config['prepare_data']['path_tabular_submit']
        # Train data
        self.data_non_temp_train: pd.DataFrame = pd.read_csv(path_non_temporal_train)
        self.n_examples_train: int = self.data_non_temp_train.shape[0]
        self.n_examples_per_file_train: int = config['prepare_data']['n_examples_per_file_train']
        # Submit data
        self.data_non_temp_submit: pd.DataFrame = pd.read_csv(path_non_temporal_submit)
        self.n_examples_submit: int = self.data_non_temp_submit.shape[0]
        self.n_examples_per_file_submit: int = config['prepare_data']['n_examples_per_file_submit']

        self.path_output: str = pathlib.Path(config['prepare_data']['path_output']).joinpath('tfrecords')
        if os.path.exists(self.path_output):
            shutil.rmtree(self.path_output)
        os.makedirs(self.path_output)
        self.feature_description = {'CGAS-CGAS_Score': tf.io.FixedLenFeature([], tf.float32), 
                                    'Physical-Height': tf.io.FixedLenFeature([], tf.float32)}

    def get_example(self, index: int, prefix: str) -> tf.train.Example:
        # This function returns an example from the raw data.
        if prefix == 'train':
            example = self.data_non_temp_train.iloc[index].to_dict()
        elif prefix == 'submit':
            example = self.data_non_temp_submit.iloc[index].to_dict()
        # TODO: use all the tabular and time series variables for writing the TFRecords
        feature = {'CGAS-CGAS_Score': self._float_feature(example['CGAS-CGAS_Score']), 
                   'Physical-Height': self._float_feature(example['Physical-Height'])}
        if 'sii' in example:
            if example['sii'].isna():  # TODO: check this NaN check
                feature['sii'] = self._float_feature(-1)
            else:
                feature['sii'] = self._float_feature(example['sii'])
        else:
            feature['sii'] = self._float_feature(-1)
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def parse_example(self, example: tf.train.Example):
        # Returns the parsed data from the input `tf.train.Example` proto.
        return tf.io.parse_single_example(example.SerializeToString(), self.feature_description)

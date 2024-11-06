import os
import shutil
import pandas as pd
from data_preparation.tfrecord_manager import TFRecordManager
import tensorflow as tf
import pathlib


class TFRecordManagerChildMind(TFRecordManager):

    def __init__(self, config):

        non_temporal_data_path = config['prepare_data']['path_tabular_train']

        self.data_non_temp: pd.DataFrame = pd.read_csv(non_temporal_data_path)
        self.n_examples: int = self.data_non_temp.shape[0]
        self.n_examples_per_file: int = config['prepare_data']['n_examples_per_file']
        self.path_output: str = pathlib.Path(config['prepare_data']['path_output']).joinpath('tfrecords')
        if os.path.exists(self.path_output):
            shutil.rmtree(self.path_output)
        os.makedirs(self.path_output)
        self.feature_description = {'CGAS-CGAS_Score': tf.io.FixedLenFeature([], tf.float32), 
                                    'Physical-Height': tf.io.FixedLenFeature([], tf.float32)}

    def get_example(self, index: int) -> tf.train.Example:
        # This function returns an example from the raw data.
        example = self.data_non_temp.iloc[index].to_dict()
        # TODO: use all the tabular and time series variables for writing the TFRecords
        feature = {'CGAS-CGAS_Score': self._float_feature(example['CGAS-CGAS_Score']), 
                   'Physical-Height': self._float_feature(example['Physical-Height'])}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def parse_example(self, example: tf.train.Example):
        # Returns the parsed data from the input `tf.train.Example` proto.
        return tf.io.parse_single_example(example.SerializeToString(), self.feature_description)

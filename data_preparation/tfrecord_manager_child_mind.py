import os
import shutil
import pandas as pd
from data_preparation.tfrecord_manager import TFRecordManager
import tensorflow as tf


class TFRecordManagerChildMind(TFRecordManager):

    def __init__(self, non_temporal_data_path: str, n_examples_per_file: int, path_output: str):
        self.data_non_temp: pd.DataFrame = pd.read_csv(non_temporal_data_path)
        self.n_examples: int = self.data_non_temp.shape[0]
        self.n_examples_per_file: int = n_examples_per_file
        self.path_output: str = path_output
        if os.path.exists(self.path_output):
            shutil.rmtree(self.path_output)
        os.makedirs(self.path_output)

    def get_example(self, index: int) -> tf.train.Example:
        # This function returns an example from the raw data.
        example = self.data_non_temp.iloc[index]
        return example

    def parse_example(self, example_proto: tf.train.Example):
        # Returns the parsed data from the input `tf.train.Example` proto.
        pass

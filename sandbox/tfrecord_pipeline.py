from typing import List
import pandas as pd
import numpy as np
import tensorflow as tf


# The following feature functions were copied from https://www.tensorflow.org/tutorials/load_data/tfrecord

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_data() -> pd.DataFrame:
    np.random.seed(0)
    data_x = np.random.random((10, 3))
    data_y = np.expand_dims(np.sqrt(np.sum(data_x * data_x, axis=1)), axis=1)
    return pd.DataFrame(np.hstack([data_x, data_y]), columns=['x', 'y', 'z', 'r'])


def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'x': _float_feature(feature0),
      'y': _float_feature(feature1),
      'z': _float_feature(feature2),
      'r': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def write_tfrecords() -> None:
    data = create_data()
    print(data)
    filename = 'output/test.tfrecord'

    with tf.io.TFRecordWriter(filename) as writer:
        for _, row in data.iterrows():
            example = serialize_example(row.x, row.y, row.z, row.r)
            writer.write(example)



feature_description = {
    'x': tf.io.FixedLenFeature([], tf.float32),  # , default_value=0.0
    'y': tf.io.FixedLenFeature([], tf.float32),  # , default_value=0.0
    'z': tf.io.FixedLenFeature([], tf.float32),  # , default_value=0.0
    'r': tf.io.FixedLenFeature([], tf.float32),  # , default_value=0.0
}


def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)


def print_tfrecords(paths: List[str]) -> None:
    parsed_dataset = tf.data.TFRecordDataset(paths).map(_parse_function)
    for parsed_record in parsed_dataset:
        print(parsed_record)
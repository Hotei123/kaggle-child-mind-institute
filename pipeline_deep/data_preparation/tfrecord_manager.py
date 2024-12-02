import os
import pathlib
import tensorflow as tf
import shutil


class TFRecordManager:
    def __init__(self, 
                 n_examples_train: int,
                 n_examples_per_file_train: int, 
                 n_examples_submit: int,
                 n_examples_per_file_submit: int, 
                 path_output: str):
        self.n_examples_train: int = n_examples_train
        self.n_examples_per_file_train: int = n_examples_per_file_train
        self.n_examples_submit: int = n_examples_submit
        self.n_examples_per_file_submit: int = n_examples_per_file_submit
        self.path_output: str = path_output

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        
    def get_example(self, index: int, prefix: str) -> tf.train.Example:
        """This function returns an example from the raw data. """
        pass

    def parse_example(self, example: tf.train.Example):
        # Returns the parsed data from the input `tf.train.Example` proto.
        pass

    def write_tfrecords(self):

        shutil.rmtree(self.path_output, ignore_errors=True)
        os.makedirs(self.path_output)

        for prefix, n_examples, n_examples_per_file in [('train', self.n_examples_train, self.n_examples_per_file_train),
                                                        ('submit', self.n_examples_submit, self.n_examples_per_file_submit)]:
            num_files = (n_examples + n_examples_per_file - 1) // n_examples_per_file

            for file_count in range(num_files):
                filename = pathlib.Path(self.path_output).joinpath(f"{prefix}_{str(file_count).zfill(len(str(num_files)) + 1)}.tfrecord")
                print(f'Writing file {filename}')
                count_start = file_count * n_examples_per_file
                count_end = min(count_start + n_examples_per_file, n_examples)

                with tf.io.TFRecordWriter(str(filename)) as writer:
                    for i in range(count_start, count_end):
                        example = self.get_example(i, prefix)
                        writer.write(example.SerializeToString())

                print(f"Saved {filename} with {count_end - count_start} examples.")

    def get_tfrecord_dataset(self, 
                             file_pattern: str,
                             num_parallel_calls: int, 
                             shuffle_buffer_size: int, 
                             batch_size: int, 
                             size_prefetch: int,
                             function_filter: callable,
                             shuffle: bool,
                             cycle_length=6,  # Number of files to read in parallel. TODO: choose this number according to host (local or remote)
                             ) -> tf.data.Dataset:
        """
        Args:
            - num_parallel_calls:  number elements to process asynchronously in parallel.
            - shuffle_buffer_size: should be a significant portion of the dataset size.
            - batch_size: should preferably take values like 128, 64, 32, 16 or 8
            - size_prefetch: should be 1-2 times the batch size, tf.data.AUTOTUNE not available in earlier versions
            - function_filter: will be used to filter the dataset for the different variants of training: i.e. cross-validation,
                               train-val-test partition, and training in full dataset. Will also be used as the identical
                               function for the case of unlabeled data.
        Returns:
            - Dataset: a `Dataset`.
        """

        if shuffle:
            files = tf.data.Dataset.list_files(str(self.path_output.joinpath(file_pattern)), shuffle=True)
            raw_dataset = files.interleave(
                tf.data.TFRecordDataset,
                cycle_length = cycle_length,
                num_parallel_calls=num_parallel_calls
            )
        else:
            files = tf.data.Dataset.list_files(str(self.path_output.joinpath(file_pattern)), shuffle=False)
            raw_dataset = tf.data.TFRecordDataset(files)

        parsed_dataset = raw_dataset.map(self.parse_example, num_parallel_calls=num_parallel_calls)
        parsed_dataset = parsed_dataset.filter(function_filter)

        if shuffle:
            parsed_dataset = parsed_dataset.shuffle(shuffle_buffer_size)

        parsed_dataset = parsed_dataset.batch(batch_size).prefetch(size_prefetch)

        return parsed_dataset
    
    @staticmethod
    def normalization_function(dataset):
        pass

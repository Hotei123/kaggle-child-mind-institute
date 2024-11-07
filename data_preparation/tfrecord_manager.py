import os
import tensorflow as tf
import shutil


class TFRecordManager:
    def __init__(self, n_examples: int, n_examples_per_file: int, path_output: str):
        self.n_examples: int = n_examples
        self.n_examples_per_file: int = n_examples_per_file
        self.path_output: str = path_output
        if os.path.exists(self.path_output):
            shutil.rmtree(self.path_output)
        os.makedirs(self.path_output)

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
        
    def get_example(self, index: int) -> tf.train.Example:
        """This function returns an example from the raw data. """
        pass

    def parse_example(self, example: tf.train.Example):
        # Returns the parsed data from the input `tf.train.Example` proto.
        pass

    def write_tfrecords(self, prefix: str, is_label: bool):
        # TODO: write labeled and unlabeled data, and use .filter to select labeled data or partitions for training, 
        #  or to write unlabeled data for submission.

        # index_len: int = len(str(int(self.n_examples / self.n_examples_per_file + 3)))
        # for count_example in range(self.n_examples):
        #     example = self.get_example(count_example)
        #     file_index = count_example // self.n_examples_per_file
        #     filename = os.path.join(self.path_output, 
        #                             f'data_{str(file_index).zfill(index_len)}.tfrecord')
        #     with tf.io.TFRecordWriter(filename) as writer:
        #         writer.write(example.SerializeToString())


        # TODO: revise the code below and carry out the above todo.
        # TODO: differentiate the cases of labeled and unlabeled data with an additional argument in method get_example,
        #  and add corresponding parameters like subsets sizes as object parameters.
        num_files = (self.n_examples + self.n_examples_per_file - 1) // self.n_examples_per_file  # Calculate number of files needed

        for file_count in range(num_files):
            filename = f"{prefix}_{file_count}.tfrecord"
            count_start = file_count * self.n_examples_per_file
            count_end = min(count_start + self.n_examples_per_file, self.n_examples)

            with tf.io.TFRecordWriter(filename) as writer:
                for i in range(count_start, count_end):
                    example = self.get_example(i)
                    writer.write(example)

            print(f"Saved {filename} with {count_end - count_start} examples.")

    def get_tfrecord_dataset(self, 
                             file_pattern: str,
                             num_parallel_calls: int, 
                             shuffle_buffer_size: int, 
                             batch_size: int, 
                             size_prefetch: int,
                             function_filter: function,
                             cycle_length=6,  # Number of files to read in parallel
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

        files = tf.data.Dataset.list_files(file_pattern)

        if file_pattern.startswith("train"):
            raw_dataset = files.interleave(
                tf.data.TFRecordDataset,
                cycle_length = cycle_length,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            raw_dataset = tf.data.TFRecordDataset(files)
     
        raw_dataset = raw_dataset.filter(function_filter)
        parsed_dataset = raw_dataset.map(self.parse_example, num_parallel_calls=tf.data.AUTOTUNE)

        if file_pattern.startswith("train"):
            parsed_dataset = parsed_dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size).prefetch(size_prefetch)
        
        return dataset

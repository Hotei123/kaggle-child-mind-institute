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
        
    def get_example(self, index: int) -> tf.train.Example:
        """This function returns an example from the raw data. """
        pass

    def parse_example(self, example_proto: tf.train.Example):
        # Returns the parsed data from the input `tf.train.Example` proto.
        pass

    def write_tfrecords(self):
        index_len: int = len(str(int(self.n_examples / self.n_examples_per_file + 3)))
        for count_example in range(self.n_examples):
            example = self.get_example(count_example)
            file_index = count_example // self.n_examples_per_file
            filename = os.path.join(self.path_output, 
                                    f'data_{str(file_index).zfill(index_len)}.tfrecord')
            with tf.io.TFRecordWriter(filename) as writer:
                writer.write(example.SerializeToString())

    def get_tfrecord_dataset(self, num_parallel_calls: int, 
                             shuffle_buffer_size: int, 
                             batch_size: int, 
                             size_prefetch: int) -> tf.data.Dataset:
        """
        Args:
            - num_parallel_calls:  number elements to process asynchronously in parallel.
            - shuffle_buffer_size: should be a significant portion of the dataset size.
            - batch_size: should preferably take values like 128, 64, 32, 16 or 8
            - size_prefetch: should be 1-2 times the batch size, tf.data.AUTOTUNE not available in earlier versions
        Returns:
            - Dataset: a `Dataset`.
        """

        raw_dataset = tf.data.TFRecordDataset(os.listdir(self.path_output))       
        parsed_dataset = raw_dataset.map(self.parse_example, num_parallel_calls=num_parallel_calls)

        dataset = (
            parsed_dataset
            .shuffle(shuffle_buffer_size)
            .batch(batch_size)
            .prefetch(size_prefetch)
        )
        
        return dataset

import tensorflow as tf
import yaml
from pipeline_deep.data_preparation.tfrecord_manager_child_mind import TFRecordManagerChildMind
from tensorflow.keras import layers, models, Input


def get_model(config):

    input_tensor_1 = Input(shape=(len(config['prepare_data']['vars_num']),))
    x1 = layers.Dense(64, activation='relu')(input_tensor_1)
    x1 = layers.Dense(32, activation='relu')(x1)

    input_tensor_2 = Input(shape=(3,))
    x2 = layers.Dense(64, activation='relu')(input_tensor_2)
    x2 = layers.Dense(32, activation='relu')(x2)
    x = layers.Concatenate()([x1, x2])

    output_tensor = layers.Dense(4, activation='softmax')(x)
    model = models.Model(inputs=[input_tensor_1, input_tensor_2], outputs=output_tensor)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


def hash_element(tensor, num_buckets=100000000):
    flattened_tensor = tf.strings.as_string(tf.reshape(tensor, [-1]))
    combined_string = tf.strings.reduce_join(flattened_tensor, separator=",")
    hashed_value = tf.strings.to_hash_bucket_fast(combined_string, num_buckets=num_buckets) / num_buckets
    return tf.cast(hashed_value, tf.float32)


def train():
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tfrec_man = TFRecordManagerChildMind(config)
    dataset_train = tfrec_man.get_tfrecord_dataset('output/tfrecords/train_*', 
                                                   6, 100, 8, 1, (lambda x, y: hash_element(x[0]) < 0.9), True)
    dataset_val = tfrec_man.get_tfrecord_dataset('output/tfrecords/train_*', 
                                                 6, 100, 8, 1, (lambda x, y: hash_element(x[0]) >= 0.9), False)
    model = get_model(config)
    model.fit(dataset_train)

    preds_train = model.predict(dataset_train)
    preds_val = model.predict(dataset_val)
    print(preds_train)    


if __name__ == '__main__':

    # TODO: calcualate metric
    # TODO: normalize data
    # TODO: add missing labels with algorithm trained in labeled data
    # TODO: add time series data
    # TODO: revise dataset parameters

    train()

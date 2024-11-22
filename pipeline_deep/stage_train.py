import numpy as np
import tensorflow as tf
import yaml
from pipeline_deep.data_preparation.tfrecord_manager_child_mind import TFRecordManagerChildMind
from tensorflow.keras import layers, models, Input
from sklearn.metrics import cohen_kappa_score


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

    n_folds = 10
    delta = 1 / n_folds
    fold_count = 1
    metrics_cv = []
    for fold_count in range(n_folds):
        tfrec_man = TFRecordManagerChildMind(config)
        dataset_train = tfrec_man.get_tfrecord_dataset('output/tfrecords/train_*', 
                                                    6, 100, 8, 1, 
                                                    lambda x, y: fold_count * delta >= hash_element(x[0]) or hash_element(x[0]) >= (fold_count + 1) * delta, 
                                                    True)
        dataset_val = tfrec_man.get_tfrecord_dataset('output/tfrecords/train_*', 
                                                    6, 100, 8, 1, 
                                                    lambda x, y: fold_count * delta < hash_element(x[0]) or hash_element(x[0]) < (fold_count + 1) * delta, 
                                                    False)
        model = get_model(config)
        model.fit(dataset_train)

        y_pred_val = np.argmax(model.predict(dataset_val), axis=1)  
        y_val = np.empty((0,))
        for x, y in dataset_val:
            y_val = np.hstack([y_val, y.numpy()])
        metrics_cv.append(cohen_kappa_score(y_pred_val, y_val, weights='quadratic'))
    print(f'CV metrics deep: {[np.round(m, 3) for m in metrics_cv]}.\nMean metric: {np.mean(metrics_cv): .3f}.')

if __name__ == '__main__':

    # TODO: calcualate metric
    # TODO: normalize data
    # TODO: add missing labels with algorithm trained in labeled data
    # TODO: add time series data
    # TODO: revise dataset parameters

    train()

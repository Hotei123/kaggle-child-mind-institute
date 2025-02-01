import pathlib
import shutil
from typing import Any, List
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from pipeline_deep.data_preparation.tfrecord_manager_child_mind import TFRecordManagerChildMind
from tensorflow.keras import layers, models, Input
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.callbacks import EarlyStopping


def get_model(shape_input_1, shape_input_2, shape_input_3, shape_input_4):

    input_tensor_1 = Input(shape=(shape_input_1,))
    x1 = layers.Dense(64, activation='relu')(input_tensor_1)
    x1 = layers.Dense(32, activation='relu')(x1)

    input_tensor_2 = Input(shape=(shape_input_2,))
    x2 = layers.Dense(64, activation='relu')(input_tensor_2)
    x2 = layers.Dense(32, activation='relu')(x2)

    input_tensor_3 = Input(shape=(shape_input_3,))
    x3 = layers.Dense(128, activation='relu')(input_tensor_3)
    x3 = layers.Dense(64, activation='relu')(x3)

    input_tensor_4 = Input(shape=(shape_input_4))
    x4 = layers.Dense(128, activation='relu')(input_tensor_4)
    x4 = layers.Dense(64, activation='relu')(x4)
    x4 = layers.Flatten()(x4)

    x = layers.Concatenate()([x1, x2, x3, x4])

    output_tensor = layers.Dense(4, activation='softmax')(x)
    model = models.Model(inputs=[input_tensor_1, input_tensor_2, input_tensor_3, input_tensor_4], outputs=output_tensor)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


def hash_element(tensor, num_buckets=100000000):
    flattened_tensor = tf.strings.as_string(tf.reshape(tensor, [-1]))
    combined_string = tf.strings.reduce_join(flattened_tensor, separator=",")
    hashed_value = tf.strings.to_hash_bucket_fast(combined_string, num_buckets=num_buckets) / num_buckets
    return tf.cast(hashed_value, tf.float32)

def get_callbacks(path_tb: str, early_stopping_monitor: str) -> List[Any]:
    if pathlib.Path(path_tb).exists():
        shutil.rmtree(path_tb, ignore_errors=True)
    early_stopping = EarlyStopping(monitor=early_stopping_monitor, patience=10, restore_best_weights=True)
    return [tf.keras.callbacks.TensorBoard(log_dir=path_tb), early_stopping]
    
def train(config, tfrecord_man: TFRecordManagerChildMind):

    batch_size = config['train']['batch_size']
    n_epochs = config['train']['n_epochs']

    # Cross-validation
    n_folds = 10
    delta = 1 / n_folds
    fold_count = 1
    metrics_cv = []
    for fold_count in range(n_folds):
        print(f'Fold {fold_count}.')
        tfrec_man = TFRecordManagerChildMind(config)
        # TODO: check batch size, and the rest of parameters for writing and reading TFRecords
        dataset_train = tfrec_man.get_tfrecord_dataset('train_*', 6, 100, 8, 1, 
                                                       lambda x, y: fold_count * delta >= hash_element(x[0]) or hash_element(x[0]) >= (fold_count + 1) * delta, 
                                                       True, True)
        dataset_val = tfrec_man.get_tfrecord_dataset('train_*', 6, 100, 8, 1, 
                                                     lambda x, y: fold_count * delta < hash_element(x[0]) and hash_element(x[0]) < (fold_count + 1) * delta, 
                                                     False, False)
        model = get_model(len(tfrecord_man.vars_num), 
                          len(tfrecord_man.vars_dummy), 
                          len(tfrecord_man.vars_time_desc), 
                          (tfrec_man.n_rows_ts, tfrec_man.n_cols_ts))
        callbacks = get_callbacks(f'output/tb_logs_{fold_count}', 'val_accuracy')
        model.fit(dataset_train, validation_data=dataset_val, 
                  epochs=n_epochs, batch_size=batch_size, callbacks=callbacks)

        y_pred_val = np.argmax(model.predict(dataset_val), axis=1)  
        y_val = np.empty((0,))
        for x, y in dataset_val:
            y_val = np.hstack([y_val, y.numpy()])
        metrics_cv.append(cohen_kappa_score(y_pred_val, y_val, weights='quadratic'))
    print(f'CV metrics deep: {[np.round(m, 3) for m in metrics_cv]}.\nMean metric: {np.mean(metrics_cv): .3f}.')

    # Training in whole dataset for submission
    tfrec_man = TFRecordManagerChildMind(config)
    # TODO: check batch size, and the rest of parameters for writing and reading TFRecords
    dataset_train_full = tfrec_man.get_tfrecord_dataset('train_*', 6, 100, 8, 1, lambda x, y: True, True, True)
    dataset_submission = tfrec_man.get_tfrecord_dataset('submit_*', 6, 100, 8, 1, lambda x, y: True, False, False)

    model = get_model(len(tfrecord_man.vars_num), 
                      len(tfrecord_man.vars_dummy), 
                      len(tfrecord_man.vars_time_desc), 
                      (tfrec_man.n_rows_ts, tfrec_man.n_cols_ts))
    callbacks = get_callbacks('output/tb_logs_full', 'accuracy')
    model.fit(dataset_train_full, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks)
    y_pred_full = np.argmax(model.predict(dataset_submission), axis=1)

    data_test_raw = pd.read_csv(tfrec_man.path_non_temporal_submit)
    submission = pd.DataFrame({'id': data_test_raw['id'], tfrec_man.var_target: y_pred_full})
    submission.to_csv(pathlib.Path(tfrec_man.path_output).parent.joinpath('submission.csv'), index=False)


if __name__ == '__main__':

    # TODO: run in Kaggle
    # TODO: use more rows of time series
    # TODO: add time series data
    # TODO: revise dataset parameters

    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tfrec_man = TFRecordManagerChildMind(config)
    train(config, tfrec_man)

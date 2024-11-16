import yaml
from data_preparation.tfrecord_manager_child_mind import TFRecordManagerChildMind


if __name__ == '__main__':

    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tfrec_man = TFRecordManagerChildMind(config)
    tfrec_man.write_tfrecords()

    dataset = tfrec_man.get_tfrecord_dataset('output/tfrecords/train_*', 6, 100, 8, 1, (lambda x: True))

    from tensorflow.keras import layers, models, Input

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
    print(model.summary())
    model.fit(dataset)

    preds = model.predict(dataset)
    print(preds)

# TODO: Use the full time series with tensorflow data.Dataset.from_generator
# TODO: Normalize the time series, possibly eliminating columns
# TODO: Separate time series into continuous parts
# TODO: Ensemble models

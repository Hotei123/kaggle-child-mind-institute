# from pipeline_steps.step_prepare_data import StepPrepareData
# from pipeline_steps.step_train import StepTrain
# import yaml

# from sandbox.dataset_minimal import train_minimal
# from sandbox.time_series import train_hard_coded


import pandas as pd
import yaml
from data_preparation.tfrecord_manager_child_mind import TFRecordManagerChildMind
from pipeline_steps.step_prepare_data import StepPrepareData
from pipeline_steps.step_train import StepTrain
from sandbox.dataset_minimal import train_minimal
from sandbox.tfrecord_pipeline import create_data, print_tfrecords, write_tfrecords
from sandbox.time_series import train_hard_coded


if __name__ == '__main__':

    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tfrec_man = TFRecordManagerChildMind(config)
    # tfrec_man.write_tfrecords()

    dataset = tfrec_man.get_tfrecord_dataset('output/tfrecords/train_*', 6, 100, 8, 1, (lambda x: True))

    # count = 0
    # for x in dataset:
    #     count += 1
    #     if count > 10:
    #         break
    #     print(x)
    #     print('\n\n\n')


    from tensorflow.keras import layers, models, Input

    # Define the input
    input_tensor = Input(shape=(2,))
    x = layers.Dense(64, activation='relu')(input_tensor)
    x = layers.Dense(32, activation='relu')(x)
    output_tensor = layers.Dense(4, activation='softmax')(x)
    model = models.Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    model.fit(dataset)

    # step_prepare_data = StepPrepareData(config)
    # step_prepare_data.export_partitions()

    # step_train = StepTrain(config)
    # step_train.train()

# TODO: Use the full time series with tensorflow data.Dataset.from_generator
# TODO: Normalize the time series, possibly eliminating columns
# TODO: Separate time series into continuous parts
# TODO: Ensemble models

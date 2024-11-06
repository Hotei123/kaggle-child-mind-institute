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
    example_0 = tfrec_man.get_example(0)
    parsed_example_0 = tfrec_man.parse_example(example_0)
    x = 0

    # step_prepare_data = StepPrepareData(config)
    # step_prepare_data.export_partitions()

    # step_train = StepTrain(config)
    # step_train.train()

# TODO: Use the full time series with tensorflow data.Dataset.from_generator
# TODO: Normalize the time series, possibly eliminating columns
# TODO: Separate time series into continuous parts
# TODO: Ensemble models

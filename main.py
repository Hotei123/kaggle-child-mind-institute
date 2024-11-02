# from pipeline_steps.step_prepare_data import StepPrepareData
# from pipeline_steps.step_train import StepTrain
# import yaml

# from sandbox.dataset_minimal import train_minimal
# from sandbox.time_series import train_hard_coded


import pandas as pd
import yaml
from pipeline_steps.step_train import StepTrain
from sandbox.dataset_minimal import train_minimal
from sandbox.tfrecord_pipeline import create_data, print_tfrecords, write_tfrecords
from sandbox.time_series import train_hard_coded


if __name__ == '__main__':

    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # step_prepare_data = StepPrepareData(config)
    # step_prepare_data.export_partitions()

    step_train = StepTrain(config)
    step_train.train()

# TODO: Use the full time series with tensorflow data.Dataset.from_generator
# TODO: Fix the fact that now only 2 variables are being passed from the CSV data
# TODO: Normalize the time series, possibly eliminating columns
# TODO: Separate time series into continuous parts
# TODO: Ensemble models

    # train_hard_coded()

    # train_minimal()

    # write_tfrecords()
    # print_tfrecords(['output/test.tfrecord'])

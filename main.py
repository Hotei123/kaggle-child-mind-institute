import yaml
from pipeline_deep.data_preparation.tfrecord_manager_child_mind import TFRecordManagerChildMind
from pipeline_deep.stage_train import train
from pipeline_shallow.stage_prepare_data import StepPrepareData
from pipeline_shallow.stage_train import StepTrain


if __name__ == '__main__':

    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tfrec_man = TFRecordManagerChildMind(config)
    tfrec_man.write_tfrecords()

    train(config, tfrec_man)

    # step_train = StepTrain(config)
    # step_train.train()

# TODO: Remove outliers
# TODO: Use the full time series with tensorflow data.Dataset.from_generator
# TODO: Normalize the time series, possibly eliminating columns
# TODO: Separate time series into continuous parts
# TODO: Ensemble models

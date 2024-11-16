import yaml
from pipeline_deep.data_preparation.tfrecord_manager_child_mind import TFRecordManagerChildMind
from pipeline_deep.stage_train import train


if __name__ == '__main__':

    train()

# TODO: Use the full time series with tensorflow data.Dataset.from_generator
# TODO: Normalize the time series, possibly eliminating columns
# TODO: Separate time series into continuous parts
# TODO: Ensemble models

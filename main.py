import yaml
from pipeline_deep.data_preparation.tfrecord_manager_child_mind import TFRecordManagerChildMind
from pipeline_deep.stage_train import train


if __name__ == '__main__':

    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    train(config)

    # tfrec_man = TFRecordManagerChildMind(config)
    # dataset_train_full = tfrec_man.get_tfrecord_dataset('train_*', 6, 100, 8, 1, lambda x, y: True, True)

# TODO: Remove outliers
# TODO: Use the full time series with tensorflow data.Dataset.from_generator
# TODO: Normalize the time series, possibly eliminating columns
# TODO: Separate time series into continuous parts
# TODO: Ensemble models

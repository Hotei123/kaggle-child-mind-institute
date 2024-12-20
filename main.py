import yaml
from pipeline_deep.data_preparation.tfrecord_manager_child_mind import TFRecordManagerChildMind
from pipeline_deep.stage_train import train
from pipeline_shallow.stage_prepare_data import StepPrepareData
from pipeline_shallow.stage_train import StepTrain


if __name__ == '__main__':

    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # tfrec_man = TFRecordManagerChildMind(config)
    # tfrec_man.write_tfrecords()

    # step_train = StepTrain(config)
    # step_train.train()

    tfrec_man = TFRecordManagerChildMind(config)
    tfrec_man.write_tfrecords()
    train(config, tfrec_man)

    # example = tfrec_man.get_example(0, 'train')
    
    # example_serialized = example.SerializeToString()
    # example_parsed = tfrec_man.parse_example(example_serialized)

    # import tensorflow as tf
    # files = tf.data.Dataset.list_files(str(tfrec_man.path_output.joinpath('train_*')), shuffle=False)
    # raw_dataset = tf.data.TFRecordDataset(files)
    # parsed_dataset = raw_dataset.map(tfrec_man.parse_example)

    # count = 0
    # for x in raw_dataset:
    #     print(count)
    #     print(tfrec_man.parse_example(x))
    #     print('\n\n\n\n')
    #     count += 1
    #     if count > 2:
    #         break
    # print(tfrec_man.parse_example(x))

    # # TODO: Write always a null array, and see if this loop runs more than 2 iterations.
    # count = 0
    # for x in parsed_dataset:
    #     print(count)
    #     count += 1
    #     if count > 2:
    #         break


# TODO: Remove outliers
# TODO: Use the full time series with tensorflow data.Dataset.from_generator
# TODO: Normalize the time series, possibly eliminating columns
# TODO: Separate time series into continuous parts
# TODO: Ensemble models

from pipeline_steps.step_prepare_data import StepPrepareData
from pipeline_steps.step_train import StepTrain
import yaml


if __name__ == '__main__':

    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    step_prepare_data = StepPrepareData(config)
    step_prepare_data.export_partitions()

    step_train = StepTrain(config)
    step_train.train()

# TODO: Parallelize preprocessing
# TODO: Use sklearn metric
# TODO: Ensemble models
# TODO: Organize Github
import pandas as pd
import xgboost as xgb

from pipeline_steps.step_prepare_data import StepPrepareData


def train():
    path_train: str = 'output/train_processed.csv'
    path_test_raw: str = 'data/child-mind-institute-problematic-internet-use/test.csv'
    path_test: str = 'output/test_processed.csv'

    data_train_xy = pd.read_csv(path_train)
    data_train_x = data_train_xy.drop(columns=['sii'])
    data_train_y = data_train_xy['sii']

    data_test_raw = pd.read_csv(path_test_raw)

    step_prepare_data = StepPrepareData()
    step_prepare_data.export_partitions()
    data_test_x = step_prepare_data.get_partition_prepared(path_test_raw, False)

    xgb_class = xgb.XGBClassifier()
    xgb_class.fit(data_train_x, data_train_y)
    y_pred_test = xgb_class.predict(data_test_x)

    submission = pd.DataFrame({'id': data_test_raw['id'], 'sii': y_pred_test})
    submission.to_csv('output/submission.csv', index=False)

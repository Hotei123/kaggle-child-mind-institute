import os
import pathlib
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
import yaml
from pipeline_shallow.stage_prepare_data import StepPrepareData


class StepTrain:

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.step_prepare_data = StepPrepareData(config)

    def train(self):
        path_output: str = self.config['prepare_data']['path_output']
        path_train_processed: str = os.path.join(path_output, 'train_processed.csv')

        data_train_xy = pd.read_csv(path_train_processed)
        data_train_x = data_train_xy.drop(columns=[self.step_prepare_data.var_target])
        data_train_y = data_train_xy[self.step_prepare_data.var_target]

        data_test_raw = pd.read_csv(self.step_prepare_data.path_tabular_test)
        data_test_x = self.step_prepare_data.get_partition_prepared(self.step_prepare_data.path_tabular_test, False)
        for col in data_train_x.columns:
            if col not in data_test_x.columns:
                data_test_x[col] = 0
        data_test_x = data_test_x[data_train_x.columns]

        model = xgb.XGBClassifier()

        data_train_x_labeled = data_train_x[data_train_y.notna()]
        data_train_x_unlabeled = data_train_x[data_train_y.isna()]
        model.fit(data_train_x_labeled, data_train_y[data_train_y.notna()])
        labels_nan_filled = model.predict(data_train_x_unlabeled)
        data_train_y.loc[data_train_y.isna()] = labels_nan_filled
        data_train_y.to_frame().to_csv(pathlib.Path(self.config['prepare_data']['path_output']).joinpath('labels_filled.csv') , index=False)

        n_folds: int = 10
        fold_size: int = int(data_train_x.shape[0] / n_folds)
        metrics_cv: List[float] = []
        print(f'Data shape: {data_train_x.shape}')
        for fold_count in range(n_folds):
            print(f'Fold {fold_count}.')
            data_train_x_fold = pd.concat([data_train_x.iloc[:fold_size * fold_count, :], 
                                           data_train_x.iloc[fold_size * (fold_count + 1):, :]])
            data_train_y_fold = pd.concat([data_train_y.iloc[:fold_size * fold_count], 
                                           data_train_y.iloc[fold_size * (fold_count + 1):]])
            data_test_x_fold = data_train_x.iloc[fold_size * fold_count: fold_size * (fold_count + 1), :]
            data_test_y_fold = data_train_y.iloc[fold_size * fold_count: fold_size * (fold_count + 1)]
            model.fit(data_train_x_fold, data_train_y_fold)
            y_pred_fold = model.predict(data_test_x_fold)
            metrics_cv.append(cohen_kappa_score(data_test_y_fold, y_pred_fold, weights='quadratic'))

        print(f'CV metrics shallow: {[np.round(m, 3) for m in metrics_cv]}.\nMean metric: {np.mean(metrics_cv): .3f}.')

        # Training in whole training data
        model.fit(data_train_x, data_train_y)
        y_pred_train = model.predict(data_train_x)
        y_pred_test = model.predict(data_test_x)

        print(f'Metric in full training data: {cohen_kappa_score(data_train_y.values, y_pred_train, weights="quadratic"): .3f}.')

        submission = pd.DataFrame({'id': data_test_raw['id'], self.step_prepare_data.var_target: y_pred_test})
        submission.to_csv(os.path.join(path_output, 'submission.csv'), index=False)


if __name__ == '__main__':
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    step_train = StepTrain(config)
    step_train.train()

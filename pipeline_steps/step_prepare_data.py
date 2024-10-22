import os
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm


class StepPrepareData:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.vars_num: List[str] = config['prepare_data']['vars_num']
        self.vars_cat: List[str] = config['prepare_data']['vars_cat']
        self.vars_cat_dummy: List[str] = None
        self.var_target: str = config['prepare_data']['var_target']
        self.path_tabular_train: str = config['prepare_data']['path_tabular_train']
        self.path_tabular_test: str = config['prepare_data']['path_tabular_test']
        self.path_output: str = config['prepare_data']['path_output']

    def get_partition_prepared(self, path: str, is_train: bool) -> pd.DataFrame:
        data: pd.DataFrame = pd.read_csv(path)
        ids: List[str] = data['id'].tolist()
        data.drop(columns=['id'], inplace=True)
        data_dummy: pd.DataFrame = pd.get_dummies(data[self.vars_cat])
        cols_select = self.vars_num + [self.var_target] if is_train else self.vars_num
        data = pd.concat([data[cols_select], data_dummy], axis=1)
        if self.vars_cat_dummy is None:
            self.vars_cat_dummy = data_dummy.columns.tolist()

        data_series = np.zeros((data.shape[0], 96))
        path_series = pathlib.Path(path).parent
        if is_train:
            path_series = path_series.joinpath('series_train.parquet')
        else:
            path_series = path_series.joinpath('series_test.parquet')
        for row_count, id_row in tqdm(enumerate(ids), total=len(ids)):
            path_child = path_series.joinpath(f'id={id_row}/part-0.parquet')
            if path_child.exists():
                data_series_row = pd.read_parquet(path_child)
                data_series_row = data_series_row.describe()
                data_series_row.drop(columns='step', inplace=True)
                data_series_row = data_series_row.values.reshape((1, -1))
                data_series[row_count, :] = data_series_row
        vars_series = [f'series_desc_{i}' for i in range(data_series.shape[1])]
        data_series = pd.DataFrame(data_series, columns=vars_series)
        data = pd.concat([data, data_series], axis=1)

        return data

    def export_partitions(self):
        for path, is_train, partition_name in zip([self.path_tabular_train, self.path_tabular_test], 
                                                  [True, False],
                                                  ['train', 'test']):
            data = self.get_partition_prepared(path, is_train)
            data.to_csv(os.path.join(self.path_output,  f'{partition_name}_processed.csv'), index=False)


if __name__ == '__main__':
    step_prepare_data = StepPrepareData()
    step_prepare_data.export_partitions()

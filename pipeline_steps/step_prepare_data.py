import pandas as pd


def export_train_test_data(path: str, is_train: bool):
    data: pd.DataFrame = pd.read_csv(path)
    if is_train:
        data.sii.fillna(0, inplace=True)
    data.drop(columns=['id'], inplace=True)
    partition_name = 'train' if is_train else 'test'
    data.to_csv(f'output/{partition_name}_processed.csv', index=False)

# path_tabular_train: str = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
path_tabular_train: str = 'data/child-mind-institute-problematic-internet-use/train.csv'
path_tabular_test: str = 'data/child-mind-institute-problematic-internet-use/test.csv'

for path_partition in zip([path_tabular_train, path_tabular_test], [True, False]):
    export_train_test_data(*path_partition)

import pandas as pd


path_tabular_train: str = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'

data_train: pd.DataFrame = pd.read_csv(path_tabular_train)
data_train.sii.fillna(0, inplace=True)
data_train.to_csv('train_processed.csv', index=False)
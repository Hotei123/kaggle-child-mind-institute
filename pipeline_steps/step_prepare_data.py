from typing import Any, List, Union
import pandas as pd


class StepPrepareData:
    def __init__(self) -> None:
        self.vars_num: List[str] = ['Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score', 'Physical-BMI', 'Physical-Height', 'Physical-Weight', 
                                    'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 'Physical-HeartRate', 
                                    'Physical-Systolic_BP', 'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins', 
                                    'Fitness_Endurance-Time_Sec', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND', 
                                    'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU', 
                                    'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR', 
                                    'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-BIA_Activity_Level_num', 
                                    'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 
                                    'BIA-BIA_FFM', 'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num', 
                                    'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW', 
                                    'PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total', 'SDS-SDS_Total_Raw', 
                                    'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday']
        self.vars_cat: List[str] = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
                                    'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 
                                    'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']
        self.vars_cat_dummy: List[str] = None
        self.var_target: str = 'sii'
        # self.path_tabular_train: str = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
        self.path_tabular_train: str = 'data/child-mind-institute-problematic-internet-use/train.csv'
        self.path_tabular_test: str = 'data/child-mind-institute-problematic-internet-use/test.csv'

    def get_partition_prepared(self, path: str, is_train: bool, 
                               use_target_nan: bool) -> pd.DataFrame:
        data: pd.DataFrame = pd.read_csv(path)
        if is_train:
            data['is_labeled'] = data[self.var_target].notna()
            if use_target_nan:
                data[self.var_target].fillna(0, inplace=True)
            else:
                data = data[data['is_labeled']]
        else:
            data['is_labeled'] = False
        data.drop(columns=['id'], inplace=True)
        data_dummy: pd.DataFrame = pd.get_dummies(data[self.vars_cat])
        cols_select = self.vars_num + [self.var_target] if is_train else self.vars_num
        cols_select.append('is_labeled')
        data = pd.concat([data[cols_select], data_dummy], axis=1)
        if self.vars_cat_dummy is None:
            self.vars_cat_dummy = data_dummy.columns.tolist()
            self.vars_cat_dummy.append('is_labeled')
        return data

    def export_partitions(self, use_target_nan: bool = True):
        for path, is_train, partition_name in zip([self.path_tabular_train, self.path_tabular_test], 
                                                  [True, False],
                                                  ['train', 'test']):
            data = self.get_partition_prepared(path, is_train, use_target_nan)
            data.to_csv(f'output/{partition_name}_processed.csv', index=False)


if __name__ == '__main__':
    # TODO: check if all variables, numerical or categorical, are included.
    # TODO: Fill Nans in training sii with predictions of algorithm trained in available sii.
    # TODO: Use time series data
    step_prepare_data = StepPrepareData()
    step_prepare_data.export_partitions()

from typing import List
import pandas as pd


class StepPrepareData:
    cols_num: List[str] = ['CGAS-CGAS_Score', 'Physical-BMI', 'Physical-Height', 'Physical-Weight', 
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

    cols_cat: List[str] = []
    col_target: str = 'sii'
    # path_tabular_train: str = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
    path_tabular_train: str = 'data/child-mind-institute-problematic-internet-use/train.csv'
    path_tabular_test: str = 'data/child-mind-institute-problematic-internet-use/test.csv'

    def export_single_partition(self, path: str, is_train: bool):
        data: pd.DataFrame = pd.read_csv(path)
        if is_train:
            data.sii.fillna(0, inplace=True)
        data.drop(columns=['id'], inplace=True)
        cols_select = self.cols_num + [self.col_target] if is_train else self.cols_num
        data = data[cols_select]
        partition_name = 'train' if is_train else 'test'
        data.to_csv(f'output/{partition_name}_processed.csv', index=False)

    def export_partitions(self):
        for path, is_train in zip([self.path_tabular_train, self.path_tabular_test], [True, False]):
            self.export_single_partition(path, is_train)


if __name__ == '__main__':

    step_prepare_data = StepPrepareData()
    step_prepare_data.export_partitions()

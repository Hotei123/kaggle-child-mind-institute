from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import xgboost as xgb

from pipeline_steps.step_prepare_data import StepPrepareData


class StepTrain:

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.step_prepare_data = StepPrepareData(config)

    @staticmethod
    def quadratic_kappa(actuals, preds, N=4):
        """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
        at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 
        of adoption rating."""
        w = np.zeros((N,N))
        O = confusion_matrix(actuals, preds, labels=range(N))
        for i in range(len(w)): 
            for j in range(len(w)):
                w[i][j] = float(((i-j)**2)/(N-1)**2)
        
        act_hist=np.zeros([N])
        for item in actuals: 
            act_hist[int(item)]+=1
        
        pred_hist=np.zeros([N])
        for item in preds: 
            pred_hist[int(item)]+=1
                            
        E = np.outer(act_hist, pred_hist)
        e_sum = E.sum()
        if e_sum > 0:
            E = E/e_sum
        o_sum = O.sum()
        if o_sum > 0:
            O = O/o_sum
        
        num=0
        den=0
        for i in range(len(w)):
            for j in range(len(w)):
                num+=w[i][j]*O[i][j]
                den+=w[i][j]*E[i][j]
        if den > 0:
            return (1 - (num/den))
        else:
            return 0

    def train(self):
        path_train: str = 'output/train_processed.csv'

        data_train_xy = pd.read_csv(path_train)
        data_train_x = data_train_xy.drop(columns=[self.step_prepare_data.var_target])
        data_train_y = data_train_xy[self.step_prepare_data.var_target]

        data_test_raw = pd.read_csv(self.step_prepare_data.path_tabular_test)
        data_test_x = self.step_prepare_data.get_partition_prepared(self.step_prepare_data.path_tabular_test, 
                                                                    False, False)
        for col in data_train_x.columns:
            if col not in data_test_x.columns:
                data_test_x[col] = 0
        data_test_x = data_test_x[data_train_x.columns]

        xgb_class = xgb.XGBClassifier()

        n_folds: int = 10
        fold_size: int = int(data_train_x.shape[0] / n_folds)
        metrics: List[float] = []
        print(f'Data shape: {data_train_x.shape}')
        for fold in range(n_folds):
            print(f'Fold {fold}.')
            data_train_x_fold = pd.concat([data_train_x.iloc[:fold_size * fold, :], 
                                           data_train_x.iloc[fold_size * (fold + 1):, :]])
            data_train_y_fold = pd.concat([data_train_y.iloc[:fold_size * fold], 
                                           data_train_y.iloc[fold_size * (fold + 1):]])
            data_test_x_fold = data_train_x.iloc[fold_size * fold: fold_size * (fold + 1), :]
            data_test_y_fold = data_train_y.iloc[fold_size * fold: fold_size * (fold + 1)]
            xgb_class.fit(data_train_x_fold, data_train_y_fold)
            y_pred_fold = xgb_class.predict(data_test_x_fold)
            metrics.append(self.quadratic_kappa(data_test_y_fold, y_pred_fold))

        print(f'CV metrics: {[np.round(m, 3) for m in metrics]}.\nMean metric: {np.mean(metrics): .3f}.')

        # Training in whole training data
        xgb_class.fit(data_train_x, data_train_y)
        y_pred_train = xgb_class.predict(data_train_x)
        y_pred_test = xgb_class.predict(data_test_x)

        print(f'Metric in full training data: {self.quadratic_kappa(data_train_y.values, y_pred_train): .3f}.')

        submission = pd.DataFrame({'id': data_test_raw['id'], self.step_prepare_data.var_target: y_pred_test})
        submission.to_csv('output/submission.csv', index=False)

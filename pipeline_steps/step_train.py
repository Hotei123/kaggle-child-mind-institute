import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import xgboost as xgb

from pipeline_steps.step_prepare_data import StepPrepareData


class StepTrain:

    step_prepare_data = StepPrepareData()
    var_target = 'sii'

    def quadratic_kappa(self, actuals, preds, N=4):
        """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
        at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 
        of adoption rating."""
        w = np.zeros((N,N))
        O = confusion_matrix(actuals, preds)
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
        E = E/E.sum()
        O = O/O.sum()
        
        num=0
        den=0
        for i in range(len(w)):
            for j in range(len(w)):
                num+=w[i][j]*O[i][j]
                den+=w[i][j]*E[i][j]
        return (1 - (num/den))

    def train(self):
        path_train: str = 'output/train_processed.csv'
        path_test_raw: str = 'data/child-mind-institute-problematic-internet-use/test.csv'

        data_train_xy = pd.read_csv(path_train)
        data_train_x = data_train_xy.drop(columns=[self.step_prepare_data.var_target])
        data_train_y = data_train_xy[self.step_prepare_data.var_target]

        data_test_raw = pd.read_csv(path_test_raw)
        data_test_x = self.step_prepare_data.get_partition_prepared(path_test_raw, False)

        xgb_class = xgb.XGBClassifier()
        xgb_class.fit(data_train_x, data_train_y)
        y_pred_train = xgb_class.predict(data_train_x)
        y_pred_test = xgb_class.predict(data_test_x)

        print(self.quadratic_kappa(data_train_y.values, y_pred_train))

        submission = pd.DataFrame({'id': data_test_raw['id'], self.step_prepare_data.var_target: y_pred_test})
        submission.to_csv('output/submission.csv', index=False)

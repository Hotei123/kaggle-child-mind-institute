### Overview

This repo corresponds to the Kaggle competition 
[Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use).

My first solution consisted of copying what most of the public solutions did: merging the non-temporal data with a .describe() of the time series data. This yielded a CV metric of 0.438 by filling the missing data with an XGBoost prediction.

In order to better use the time series data, I wrote all the data into TFRecords using multi-threading, and trained a Keras model with it.

### DVC Pipelines

The file `dvc.yaml` contains two pipelines: one for shallow and one for deep learning. To execute the pipelines, just run `dvc repro`, and will be executed the pipelines' stages that have changed. For printing the Directed Acyclic Graph (DAG) corresponding to the pipelines, run `dvc dag`, which should return something like this:

```
+----------------------+ 
| prepare_data_shallow | 
+----------------------+ 
            *            
            *            
            *            
    +---------------+    
    | train_shallow |    
    +---------------+    
+-------------------+  
| prepare_data_deep |  
+-------------------+  
          *            
          *            
          *            
    +------------+     
    | train_deep |     
    +------------+
```     

### Metrics in each iteration (shallow)

- Using only numerical variables:
CV metrics: [0.304, 0.342, 0.448, 0.385, 0.403, 0.309, 0.353, 0.396, 0.258, 0.341].
Mean metric:  0.354.
Metric in full training data:  0.990.

- After adding integer variables:
CV metrics: [0.345, 0.357, 0.398, 0.39, 0.404, 0.332, 0.398, 0.383, 0.301, 0.333].
Mean metric:  0.364.
Metric in full training data:  0.991.

- After adding dummy variables:
CV metrics: [0.364, 0.335, 0.421, 0.411, 0.329, 0.342, 0.361, 0.374, 0.376, 0.352].
Mean metric:  0.366.
Metric in full training data:  0.994.

- Filling unlabeled nans with algorithm's predictions:
CV metrics: [0.42, 0.411, 0.485, 0.49, 0.442, 0.318, 0.493, 0.499, 0.414, 0.409].
30.9s	18	Mean metric:  0.438.

### Metrics in each iteration (deep)

- CV metrics deep: [0.264, 0.271, 0.249, 0.29, 0.222, 0.103, 0.225, 0.215, 0.314, 0.033].
Mean metric:  0.219.

- After normalizing: [-0.005, 0.252, 0.265, 0.101, 0.345, 0.222, 0.368, 0.217, 0.015, 0.276].
Mean metric: 0.2056.

- After using filled NaNs labels: [0.151, 0.186, 0.252, 0.269, 0.265, 0.364, 0.351, 0.265, 0.248, 0.32].
Mean metric:  0.267.

- After using dummy vars: [0.25, 0.195, 0.248, 0.475, 0.192, 0.4, 0.273, 0.235, 0.246, 0.151].
Mean metric:  0.267.

- After using pd.describe() vars: [0.156, 0.038, 0.245, 0.421, 0.297, 0.289, 0.281, 0.141, 0.366, 0.283].
Mean metric:  0.252

- After using normalization in series of size 10, and more epochs: [0.205, 0.269, 0.27, 0.376, 0.432, 0.444, 0.285, 0.353, 0.335, 0.44]. Mean metric: 0.341.

- Series of size 10, and 100 epochs: [0.287, 0.267, 0.353, 0.304, 0.458, 0.424, 0.349, 0.374, 0.362, 0.349].
Mean metric: 0.353.

### Some comments on normalization

The reason behind normalizing the tf.data.Dataset during training time and not during writing the TFRecords, is
because the subset with respect the normalization is taken changes, because it can be a fold in the cross-validation,
or can be the full training set for training for the final submission. That's why I preferred a little of overhead
in favour of writing code easier.
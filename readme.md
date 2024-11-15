### Overview

This repo corresponds to the Kaggle competition 
[Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use).

My first solution consisted of copying what most of the public solutions did: merging the non-temporal data with a .describe() of the time series data. This yielded a CV metric of 0.438 by filling the missing data with an XGBoost prediction.

In order to better use the time series data, I wrote all the data into TFRecords using multi-threading, and trained a Keras model with it.

TODO: describe how to run the shallow and deep solutions
TODO: delete the sections of this file that are not going to be published.
TODO: finish the TFRecord writing and training

### DVC Pipelines

The data preparation and training are different for the cases of shallow and deep learning, because in the latter I have to use TFRecord files, and a different training code. That's why there are two DVC pipelines: one for shallow models and one for deep learning models.

For running the DVC pipeline for shallow training, run `dvc repro pipeline_steps_shallow/dvc.yaml`. For the case of the Keras (deep learning) training, run `dvc repro pipeline_steps_shallow/dvc.yaml`.

Having DVC pipelines not in the root of the project has disadvantages, because paths will have to be changed if you want to execute commands from the root of the project to execute files in such root. This can be mandatory for debugging in VSCode.

### Metrics in each iteration:

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

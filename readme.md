### Overview

This repo corresponds to the Kaggle competition 
[Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use).

My first solution consisted on copying what most of the public solutions did: merging the non-temporal data with a .describe() of the time series data. This yielded a CV metric of 0.438 by filling the missing data with an XGBoost prediction.

In order to better use the time series data, I wrote all the data into TFRecords, and trained a Keras model with it.

TODO: describe how to run the shallow and deep solutions
TODO: delete the sections of this file that are not going to be published.

### Install

Working:
- numpy=1.19.5
- tensorflow-gpu=2.2.0
- pandas=1.2.0

These could work:
- pyarrow=17.0.0
- fastparquet=0.4.1

### Fix

- pandas, pyarrow and fastparquet are not compatible

### Metrics in each iteration:

Metrics with only numerical variables:
CV metrics: [0.304, 0.342, 0.448, 0.385, 0.403, 0.309, 0.353, 0.396, 0.258, 0.341].
Mean metric:  0.354.
Metric in full training data:  0.990.

Metrics after adding integer variables:
CV metrics: [0.345, 0.357, 0.398, 0.39, 0.404, 0.332, 0.398, 0.383, 0.301, 0.333].
Mean metric:  0.364.
Metric in full training data:  0.991.

Metrics after adding dummy variables:
CV metrics: [0.364, 0.335, 0.421, 0.411, 0.329, 0.342, 0.361, 0.374, 0.376, 0.352].
Mean metric:  0.366.
Metric in full training data:  0.994.

Filling unlabeled nans with algorithm's predictions:
CV metrics: [0.42, 0.411, 0.485, 0.49, 0.442, 0.318, 0.493, 0.499, 0.414, 0.409].
30.9s	18	Mean metric:  0.438.

CV metrics: [0.37, 0.449, 0.446, 0.508, 0.46, 0.408, 0.47, 0.489, 0.378, 0.373].
Mean metric:  0.435.
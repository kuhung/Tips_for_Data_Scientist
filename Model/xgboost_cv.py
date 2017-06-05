# Created on: 2017/06/05
# Source: lost



import xgboost as xgb
import pandas as pd
import numpy as np

labels = train["label"]

features = train.drop(['label','id'], axis=1)
data_dmat = xgb.DMatrix(data=features, label=labels)


params={}

rounds = 50

result = xgb.cv(params=params, dtrain=data_dmat, num_boost_round=rounds, early_stopping_rounds=5, as_pandas=True, seed=2333)
print result

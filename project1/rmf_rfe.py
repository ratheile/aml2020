import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, \
  GradientBoostingRegressor 

from sklearn.feature_selection import RFE, RFECV

from sklearn.linear_model import \
    LinearRegression, \
    Lasso, \
    Ridge, \
    ElasticNet

import lightgbm as lgbm
from xgboost import XGBRegressor

def rfe_dim_reduction(X,y,method,estimator, estimator_args, min_feat=20, step=10, verbose=1):
  # Good read: https://scikit-learn.org/stable/modules/feature_selection.html
  # Also: https://www.datacamp.com/community/tutorials/feature-selection-python
  # Different types of feature selection methods:
  # 1. Filter methods: apply statistical measures to score features (corr coef and Chi^2).
  # 2. Wrapper methods: consider feature selection a search problem (e.g. RFE)
  # 3. Embedded methods: feature selection occurs with model training (e.g. LASSO)
  
  estimator_dic = {
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'lightgbm': lgbm.LGBMRegressor(boosting_type='dart'),
    'elacticnet': ElasticNet(),
    'xgboost': XGBRegressor(**estimator_args)
  }
  estimator = estimator_dic[estimator] # TODO: this is an arbitrary choice and the result is influenced by this!
  if method == "rfe":
    selector = RFE(estimator, n_features_to_select=min_feat, step=step, verbose=verbose)
  elif method == "rfecv":
    selector = RFECV(estimator, step=step, cv=5, verbose=verbose, min_features_to_select=min_feat)
  # TODO: consider other methods? e.g. tree-based feature selection + SelectFromModel?

  selector = selector.fit(X, y.values.ravel()) # Transformation in y as requested by function
  print(f'Original number of features : {X.shape[1]}')
  print(f"Final number of features : {selector.n_features_}")
  X_red = selector.transform(X)
  X_red = pd.DataFrame(X_red, columns=X.columns[selector.support_])

  return X_red
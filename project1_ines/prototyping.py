# Prototyping script
#%% Imports
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
import pandas as pd
import numpy as np
import logging

import os
repopath = '/Users/inespereira/Documents/Github/aml2020'
os.chdir(repopath)

from project1.main import \
  fill_nan, \
  find_isolation_forest_outlier

from project1_ines.main import \
  find_isolation_forest_outlier

#%% Load training dataset from csv
X = pd.read_csv(f'{repopath}/project1_ines/X_train.csv')
X = X.iloc[:,1:]
y = pd.read_csv(f'{repopath}/project1_ines/y_train.csv')
y = y.iloc[:,1:]
X_test = pd.read_csv(f'{repopath}/project1_ines/X_test.csv')
logging.info('I have imported your training dataset! :D')
print(f'Shape of training set is {X.shape}')
print(X)

#%%
print(X_test)

#%% Fill nan
X[:] = fill_nan(X,'median')

#%% Look at X again
X = pd.DataFrame(data=X)
X.describe

#%% Remove outliers
def find_isolation_forest_outlier(X,y,cont_lim):
  clf = IsolationForest(contamination=cont_lim).fit(X)
  y_pred_train = clf.predict(X)
  outliers = np.array(np.where(y_pred_train==-1))
  print(f"Total number of outliers removed: {outliers.size}")
  outliers = outliers.tolist()
  outliers = [ item for elem in outliers for item in elem]
  X_inliers = X.drop(index=outliers)
  y_inliers = y.drop(index=outliers)
  return X_inliers, y_inliers

cont_lim = 0.1
X, y = find_isolation_forest_outlier(X,y,cont_lim)

#%% Look at data again
print(f'After outlier removal, training set shape is {X.shape}')
print(f"Training set labels' shape is {y.shape}")

#%% Feature selection
def rfe_dim_reduction(X,y,method,estimator):
  # Good read: https://scikit-learn.org/stable/modules/feature_selection.html
  # Also: https://www.datacamp.com/community/tutorials/feature-selection-python
  # Different types of feature selection methods:
  # 1. Filter methods: apply statistical measures to score features (corr coef and Chi^2).
  # 2. Wrapper methods: consider feature selection a search problem (e.g. RFE)
  # 3. Embedded methods: feature selection occurs with model training (e.g. LASSO)

  estimator = estimator # TODO: this is an arbitrary choice and the result is influenced by this!
  if method == "rfe":
    selector = RFE(estimator, n_features_to_select=60, step=10, verbose=1)
  elif method == "rfecv":
    selector = RFECV(estimator, step=10, cv=5, verbose=1, min_features_to_select=20)
  # TODO: consider other methods? e.g. tree-based feature selection + SelectFromModel?

  selector = selector.fit(X, y)
  print('Original number of features : %s' % X.shape[1])
  print("Final number of features : %d" % selector.n_features_)
  X_red = selector.transform(X)
  X_red = pd.DataFrame(X_red, columns=X.columns[selector.support_])

  return X_red, selector

#%% Run feature selection
estimator = lgbm.LGBMRegressor()
X, selector = rfe_dim_reduction(X,y,"rfe", estimator)

#%% Look at the new X and keep it
X_red = X
X.describe
print(X_red)
X.shape
#%% Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# min_max_scaler.fit_transform(X)
normData = pd.DataFrame(min_max_scaler.fit_transform(X), index=X.index, columns=X.columns)

#%% Print normalized dataset
print(normData)
#%% Decide later whether you want to include this.
# # Reference: https://www.scikit-yb.org/en/latest/api/model_selection/rfecv.html
# from yellowbrick.model_selection import RFECV as RFECV2
# model = Ridge()
# visualizer = RFECV2(model,step=20)
# visualizer.fit(X, y)        # Fit the data to the visualizer
# visualizer.show()           # Finalize and render the figure

#%% TRain test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.05)

#%%
print(X_train)
# %%
y.values.ravel()
y.shape
# %%

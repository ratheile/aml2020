import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import logging

def fill_nan(X, method):
  if method == 'mean':
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
  elif method == 'median':
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
  # elif method == 'freq': # TODO: consider whether we want this option as well.
  #   X = X.apply(lambda col: col.fillna(col.mode()), axis=0)
  return X

def outlier_detection(X,y,method):
  if method == 'isol_forest':
    clf = IsolationForest(random_state=0).fit(X)
    y_pred_train = clf.predict(X)
    # inliers = np.array(np.where(y_pred_train==1))
    outliers = np.array(np.where(y_pred_train==-1))
    # print(f"Here are a few examples of inliers:\n{inliers}")
    # print(f"The total number of inliers is: {inliers.size}")
    # print(f"Here are the detected outliers:\n{outliers}")
    print(f"The total number of outliers removed: {outliers.size}")
    outliers = outliers.tolist()
    outliers = [ item for elem in outliers for item in elem]
    X_inliers = X.drop(index=outliers)
    y_inliers = y.drop(index=outliers)
    return X_inliers, y_inliers

def feature_selection(X,y,method):
  # Good read: https://scikit-learn.org/stable/modules/feature_selection.html
  # Also: https://www.datacamp.com/community/tutorials/feature-selection-python
  # Different types of feature selection methods:
  # 1. Filter methods: apply statistical measures to score features (corr coef and Chi^2).
  # 2. Wrapper methods: consider feature selection a search problem (e.g. RFE)
  # 3. Embedded methods: feature selection occurs with model training (e.g. LASSO)

  estimator = Ridge() # TODO: this is an arbitrary choice and the result is influenced by this!
  if method == "rfe":
    selector = RFE(estimator, n_features_to_select=20, step=10, verbose=0)
  elif method == "rfecv":
    selector = RFECV(estimator, step=1, cv=5, verbose=0, min_features_to_select=20)
  # TODO: consider other methods? e.g. tree-based feature selection + SelectFromModel?

  selector = selector.fit(X, y)
  print('Original number of features : %s' % X.shape[1])
  print("Final number of features : %d" % selector.n_features_)
  X_red = selector.transform(X)
  X_red = pd.DataFrame(X_red)

  return X_red, selector


def run(run_cfg, env_cfg):
  logging.warn(env_cfg)

  # Load training dataset from csv
  datapath = env_cfg['datasets/project1/path']
  X = pd.read_csv(datapath+'/X_train.csv')
  X = X.iloc[:,1:]
  y = pd.read_csv(datapath+'/y_train.csv')
  y = y.iloc[:,1:]
  logging.info('Training dataset imported')

  # Subtask 0: filling up NaNs
  X = fill_nan(X, run_cfg['preprocessing/nan'])
  logging.warn("Filling up NaNs with mean ... done")

  # Subtask 1: Outlier detection
  if run_cfg['preprocessing/outlier']:
    X,y = outlier_detection(X,y,run_cfg['preprocessing/outlier_type'])
    logging.warn("outlier detection ... done")

  #Subtask 2: Feature selection
  if run_cfg['preprocessing/dim_red']:
    # Overwrites X with reduced version of X
    X, selector = feature_selection(X,y,run_cfg['preprocessing/dim_red_type'])
    logging.warn("dimensionality reduction ... done")

  ############### UNUSED CODE ###############
  # both syntax work 
  # ['outer']['inner']
  # outer/inner
  # the first is primarely used if you iterate over things
  for i in run_cfg['array']:
    # seriously tho: never use this as a var name because of self ref!!!
    logging.error(f'this: {i["this"]} and that: {i["that"]}')

  # invoke different functions
  options_dict = {
  'option1' : option1,
  'option2' : option2
  }
  result_f = options_dict[run_cfg['task/variant']]
  logging.info(result_f(run_cfg['task']['cfg'])) 
  logging.info(result_f(run_cfg['task/cfg'])) 
  
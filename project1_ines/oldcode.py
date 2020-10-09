import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.decomposition import PCA
import logging

def fill_nan(X, method):
  if method == 'mean':
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
  elif method == 'median':
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
  # elif method == 'freq': # TODO: consider whether we want this option as well.
  #   X = X.apply(lambda col: col.fillna(col.mode()), axis=0)
  return X

def run(run_cfg, env_cfg):

    # Load training dataset from csv
    datapath = env_cfg['datasets/project1/path']
    X = pd.read_csv(datapath+'/X_train.csv')
    X = X.iloc[:,1:]
    y = pd.read_csv(datapath+'/y_train.csv')
    y = y.iloc[:,1:]
    logging.info('Training dataset imported')

    # Inspect dataset
    X_stats = X.describe()
    X_stats.loc[len(X_stats.index)] = X.median(axis=0)
    X_stats = X_stats.rename(index={8: 'median'})

    # Why not just PCA and then use the first ... PC to make predictions?
    # Eliminate correlated features
    # Train one simple model and discover irrelevant features
    # Outlier detection

    # This immediately shows:
    # - missing data (NaN in the dataset): dropping all rows with NaNs yield 0 observations:
    # df = X.dropna(axis=1) # we loose all observations
    # df = X.dropna(axis=0) # we lose all features
    #    - This means we need to find a way to deal with them!
    # X = X.apply(lambda row: row.fillna(row.mean()), axis=1)  # fills NaNs with row mean
    # - widely different features

    X_min = X.min(axis=1)
    min_all = X_min.min()
    X_max = X.max(axis=1)
    max_all = X_max.max()
    n_bins = 10
    # X['mean'] = X.mean(axis=1)
    # X['median'] = X.median(axis=1)

    # Bin the data and create a secondary dataset
    # X_sec = X.apply(lambda row: row.cut(x=row, bins=np.linspace(min_all, max_all, n_bins), labels=list(range(n_bins))), axis=1)
    X_sec = X
    for i in range(X.shape[0]):
        X_sec.iloc[i, :] = pd.cut(x=X.iloc[i, :], bins=np.linspace(min_all, max_all, n_bins), labels=list(range(n_bins-1)))


    # General strategy will need:
    # - assume what you have is the MRI signal intensity for some region. Since you expect cortical atrophy and ventricular
    # enlargement with increased age, do a histogram of the values. Define number of bins as a parameter.
    # - feature selection (manual versus automatic? We don't know what the features in the dataset actually are...

    # Questions
    # - Can I assume these features measure the same thing? If so, the histogram thing works, otherwise it does not.

    ########################################
    #   START SIMPLE - LINEAR REGRESSION   #
    ########################################

    # Fit model using data as is with NaNs filled in with row mean.
    X = X.apply(lambda row: row.fillna(row.mean()), axis=1)  # fills NaNs with row mean
    reg = LinearRegression().fit(X, y)
    reg.score(X, y)
    reg.coef_
    reg.intercept_

    # Same thing but with Lasso
    reg_lasso = Lasso(alpha=0.01)
    reg_lasso = reg_lasso.fit(X, y)
    reg_lasso.score(X, y)  # worse than simple linear model
    reg_lasso.coef_
    reg_lasso.intercept_

    # Same thing but with Ridge regression
    reg_ridge = Ridge(alpha=0.01)
    reg_ridge = reg_ridge.fit(X, y)
    reg_ridge.score(X, y)  # same as simple linear regression - might as well use this
    reg_ridge.coef_
    reg_ridge.intercept_


    # Try PCA
    pca = PCA(n_components=10)
    pca.fit(X)




def run(run_cfg, env_cfg):
  logging.warn(env_cfg)

  if run_cfg['preprocessing/nan']:
    logging.warn("outlier detection ... done")

  if run_cfg['preprocessing/outlier']:
    logging.warn("outlier detection ... done")

  if run_cfg['preprocessing/dim_red']:
    logging.warn("dimensionality reduction ... done")
  
  # both syntax work 
  # ['outer']['inner']
  # outer/inner
  # the first is primarely used if you iterate over things
  for i in run_cfg['array']:
    # seriously tho: never use this as a var name because of self ref!!!
    logging.error(f'this: {i["this"]} and that: {i["that"]}')

  # invoke different functions
  result_f = options_dict[run_cfg['task/variant']]
  logging.info(result_f(run_cfg['task']['cfg'])) 
  logging.info(result_f(run_cfg['task/cfg'])) 
  
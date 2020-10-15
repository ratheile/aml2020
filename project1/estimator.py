import logging
import os
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .main import preprocess

from .rmf_ffu import ffu_dim_reduction, \
  drop_feat_cov_constant

from .rmf_rfe import rfe_dim_reduction

from .outlier import remove_isolation_forest_outlier, \
  find_iso

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import RFE, RFECV

from sklearn.ensemble import IsolationForest, \
  GradientBoostingRegressor

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import \
    RepeatedKFold, \
    cross_val_score, \
    train_test_split

from sklearn.linear_model import \
    LinearRegression, \
    Lasso, \
    Ridge, \
    ElasticNet
import logging

class Project1Estimator(BaseEstimator):

  ##################### Estimator API ########################
  def __init__(self, run_cfg, env_cfg):
    self.env_cfg = env_cfg
    self.run_cfg = run_cfg
    logging.info('Estimator initialized')

  def fit(self, X, y):
    # run(self.run_cfg, self.env_cfg)
    logging.info('fit')

  def predict(self, X):
    check_is_fitted(self)
    logging.info('predict')
    X, X_u, y = preprocess(run_cfg, X, X_u, y)

    # Reduce dimensionality of test dataset based on feature selection on training data 
    X_u = X_u[X_train.columns]

    for t_name in tasks:
      model_dict = estimators[t_name]
      model = model_dict['model']() # factory
      fit_f = model_dict['fit'](model,X_train,y_train)
      logging.info(model.score(X_test, y_test))
      y_u = model.predict(X_u)
      if len(y_u.shape) > 1:
        yuf = y_u.flatten()
        y_u = yuf
      y_u_df =  pd.DataFrame({
        'id': np.arange(0,len(y_u)).astype(float),
        'y': y_u
      })

      if not os.path.exists('predictions'):
        os.makedirs('predictions')
      y_u_df.to_csv(f'predictions/{t_name}_y.csv', index=False)

  ##################### Cross Validation ########################
  def cross_validate(X, y):
    tasks = run_cfg['tasks']
    estimators = cfg_to_estimators(run_cfg)

    X_train, X_test, y_train, y_test = train_test_split(
      X,y ,test_size=run_cfg['overfit/test_size'])

    task_args = [{
      'X': X_train.copy(deep=True),
      'y': y_train.copy(deep=True),
      'task': t
    } for t in tasks]
    args = [{'estimators':estimators,'task_args': a} for a in task_args]

    train_scores = []
    for i, arg in enumerate(args):
      s, m = pool_f(arg)
      train_scores.append(s)

    train_scores_mean = pd.DataFrame( np.array(train_scores).T).mean()
    train_scores_mean.index = run_cfg['tasks']
    logging.info(train_scores_mean)


  ##################### Custom functions ########################
  def pca_dim_reduction(self, X, n_comp):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    logging.info(f"\nPC 1 with scaling:\n { pca.components_[0]}")
    return X

  # def normalize(X,X_test,method):
  #   scaler_dic = {
  #     'minmax': MinMaxScaler(),
  #     'standard': StandardScaler(),
  #   }
  #   scaler = scaler_dic[method]
  #   scaler = scaler.fit(X)
  #   X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
  #   X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
  #   return X_scaled, X_test_scaled

  scaler_dic = {
    'minmax': MinMaxScaler(),
    'standard': StandardScaler(),
  }

  def normalize(self, X,method):
    scaler = scaler_dic[method]
    scaler = scaler.fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    return X_scaled


  def autofeat_dim_reduction(self, X,y):
    fsel = FeatureSelector(verbose=1)
    X = fsel.fit_transform(X,y)
    return X

  def simple_fit(self, model, X, y):
    model = model.fit(X, y)
    return model 

  def auto_crossval(self, model, X, y):
    rkf = RepeatedKFold(n_splits=10, n_repeats=2)

    scores = cross_val_score(
      model, X, y, cv=rkf, verbose=1,
      scoring='r2'
    )
    return scores

  def cfg_to_estimators(self, run_cfg):
    elasticnet_cfg = run_cfg['models/elasticnet']
    lasso_cfg = run_cfg['models/lasso']
    ridge_cfg = run_cfg['models/ridge']
    estimators = {
      'elasticnet': {
        'model': lambda: ElasticNet(alpha=1.01),
        'fit': self.simple_fit,
        'crossval_fit': lambda m,X,y: auto_crossval(m,X,y),
        'validate': lambda m,X,y: m.score(m,X,y)
      },
      'lasso': {
        'model': lambda: Lasso(alpha=1.01),
        'fit': self.simple_fit,
        'crossval_fit': lambda m,X,y: auto_crossval(m,X,y),
        'validate': lambda m,X,y: m.score(m,X,y)
      },
      'ridge': {
        'model': lambda: Ridge(alpha=1.01),
        'fit': self.simple_fit,
        'crossval_fit': lambda m,X,y: auto_crossval(m,X,y),
        'validate': lambda m,X,y: m.score(m,X,y)
      },
      'lightgbm': {
        'model': lambda : lgbm.LGBMRegressor(),
        'fit': self.simple_fit,
        'crossval_fit': lambda m,X,y: auto_crossval(m,X,y),
        'validate': lambda m,X,y: m.score(m,X,y)
      }
    }
    return estimators

  def pool_f(self, args):
    estimators = args['estimators']
    task_args = args['task_args']
    model_dict = estimators[task_args['task']]
    model = model_dict['model']() # factory
    crossval_fit = model_dict['crossval_fit']

    # run the task:
    X = task_args['X']
    y = task_args['y']
    return (crossval_fit(model, X, y), model)


  def fill_nan(self, X, strategy):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    return(imputer.fit_transform(X))

  def preprocess(self, run_cfg, X, X_u, y):

    # Remove NaN from training and test data
    X[:] = fill_nan(X, run_cfg['preproc/imputer/strategy'])  # train
    X_u[:] = fill_nan(X_u, run_cfg['preproc/imputer/strategy'])  # test

    # Run first loop on drop_feat_cov_constant to remove
    # features with 0 mean and constant signal
    # cvmin should be in the range 1e-4 for this task
    if run_cfg['preproc/zero_and_const/enabled']:
      X = drop_feat_cov_constant(X, run_cfg['preproc/zero_and_const/cvmin'])
    X_u = X_u[X.columns]  # Drop these columns in the test set as well

    # Remove outliers (rows/datapoints)
    if run_cfg['preproc/outlier/enabled']:
      cont_lim =  run_cfg['preproc/outlier/cont_lim']
      if run_cfg['preproc/outlier/impl'] == 'ines':
        X,y = find_isolation_forest_outlier(X,y, cont_lim)
      else:
        X = remove_isolation_forest_outlier(X,y, cont_lim)

    # # Normalization training and test data
    flag_normalize = run_cfg['preproc/normalize/enabled']

    # if flag_normalize: 
    #   X, X_u = normalize(X, X_u, run_cfg['preproc/normalize/method'])

    # Normalize TODO: Check this out
    X = normalize(X, run_cfg['preproc/normalize/method'])
    X_u = normalize(X_u, run_cfg['preproc/normalize/method'])

    # Reduce data set dimensionality
    rfe_method = run_cfg['preproc/rmf/rfe/method']
    rfe_estimator = run_cfg['preproc/rmf/rfe/estimator']
    rmf_pipelines = {
      'ffu': lambda X,y: ffu_dim_reduction(run_cfg,X,y),
      'rfe': lambda X,y: rfe_dim_reduction(X,y,rfe_method, rfe_estimator),
      'auto' : lambda X,y: autofeat_dim_reduction(X,y)
    }
    rmf_pipeline_name = run_cfg['preproc/rmf/pipeline']
    X = rmf_pipelines[rmf_pipeline_name](X,y)

    # Apply pca (with min max normalization)
    if run_cfg['preproc/rmf/pca/enabled']:
      if not flag_normalize: 
        logging.error('Unnormalized data as PCA input!')
      n_comp = run_cfg['preproc/rmf/pca/n_comp']
      X = pca_dim_reduction(X, n_comp)

    return X, X_u, y



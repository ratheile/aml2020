import logging
import os
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from project2.rmf_ffu import ffu_dim_reduction, \
  drop_feat_cov_constant

from project2.rmf_rfe import rfe_dim_reduction

from project2.outlier import find_isolation_forest_outlier

from autofeat import FeatureSelector

from sklearn.metrics import r2_score

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import RFE, RFECV

from sklearn.ensemble import IsolationForest, \
  GradientBoostingRegressor, \
  StackingClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import \
    RepeatedKFold, \
    cross_val_score, \
    train_test_split, \
    StratifiedKFold

from sklearn.linear_model import \
    LinearRegression, \
    Lasso, \
    Ridge, \
    ElasticNet

import hashlib
import logging
import json
import time

import lightgbm as lgbm
from xgboost import XGBRegressor

class Project2Estimator(BaseEstimator):

  ##################### Estimator API ########################
  def __init__(self, run_cfg, env_cfg, slice_cfg=None, **args):
    self.env_cfg = env_cfg
    self.run_cfg = run_cfg
    self.slice_cfg = slice_cfg
    logging.info(f'Estimator initialized {args}')

    self.scaler_dic = {
      'minmax': lambda: MinMaxScaler(),
      'standard': lambda: StandardScaler(),
    }
    
    self.estimators = self.cfg_to_estimators(run_cfg)

    if slice_cfg is not None:
      self.parameters = list(slice_cfg['run_cfg'].keys())
    else:
      self.parameters = []


  def fit(self, X, y, X_test):
    begin_time = time.time()
    
    # Preprocessing
    X = self.df_sanitization(X)
    y = self.df_sanitization(y)

    hash_dir = self.env_cfg['datasets/project2/hash_dir']
    # Bypass data input
    skip_preprocessing = False

    df_hash_f = lambda df: hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()
    fn_func = lambda hash_df, hash_cfg: f'{hash_dir}/{hash_df}_{hash_cfg}.pkl'
    
    load_flag = self.run_cfg['persistence/load_from_file']
    save_flag = self.run_cfg['persistence/save_to_file']

    if load_flag or save_flag:
      cfg_hash = hashlib.sha256(json.dumps(self.run_cfg['preproc']).encode()).hexdigest()
      X_hash = df_hash_f(X)
      y_hash = df_hash_f(y)
      
    if load_flag:
      X_file = fn_func(X_hash,cfg_hash)
      y_file = fn_func(y_hash,cfg_hash)

      if os.path.isfile(X_file) and os.path.isfile(y_file):
        logging.warning(f'found pickle for X: {X_file}')
        logging.warning(f'found pickle for y: {y_file}')
        X = pd.read_pickle(X_file)
        y = pd.read_pickle(y_file)
        skip_preprocessing = True

    if not skip_preprocessing:
      X, y, X_test = self.preprocess(self.run_cfg, X, y, X_test)
      # X, y = check_X_y(X, y) # TODO: returns wierd stuff

    if save_flag and not skip_preprocessing:
      X.to_pickle(fn_func(X_hash,cfg_hash))
      y.to_pickle(fn_func(y_hash,cfg_hash))


    # Regression model fit
    estimator_name = self.run_cfg['fit_model']
    estimator_cfg = self.estimators[estimator_name]
    model = estimator_cfg['model']() # factory
    fitted_model = estimator_cfg['fit'](model, X, y)

    end_time = time.time()
    logging.info(f'Fitting completed in: {end_time - begin_time:.4f} seconds.')

    # store
    self._fitted_model_ = fitted_model
    self._X = X
    self._y = y
    self._X_test = X_test


  def predict(self, X_u):

    check_is_fitted(self)

    X_u = self.df_sanitization(X_u)
    X_u = self.preprocess_unlabeled(self.run_cfg, X_u)

    # Reduce dimensionality of test dataset
    # based on feature selection on training data 
    X_u = X_u[self._X.columns]
    y_u = self._fitted_model_.predict(X_u)
    return y_u


  def score(self, X, y=None):
    return(r2_score(self.predict(X), y))


  def get_params(self, deep=True):
    out = {}
    out['run_cfg'] = self.run_cfg
    out['env_cfg'] = self.env_cfg

    if self.slice_cfg is not None:
      out['slice_cfg'] = self.slice_cfg

    for key in self.parameters:
      out[key] = self.run_cfg[key]
    return out


  def set_params(self, **params):
    if not params:
      # Simple optimization to gain speed (inspect is slow)
      return self
    
    if 'run_cfg' in params:
      logging.warn('run_cfg set in set_params')
      self.run_cfg = params['run_cfg']

    if 'env_cfg' in params:
      logging.warn('env_cfg set in set_params')
      self.env_cfg = params['env_cfg']

    if 'slice_cfg' in params:
      logging.warn('slice_cfg set in set_params')
      self.slice_cfg = params['slice_cfg']

      logging.warn(f'updating params: {params}')
    for key, value in params.items():
      if key != 'run_cfg' and key != 'env_cfg' and key != 'slice_cfg':
        self.run_cfg[key] = value
    
    return self

  ##################### Cross Validation ########################
  def cv_task(self, args):
    estimators = args['estimators']
    task_args = args['task_args']
    model_dict = estimators[task_args['task']]
    model = model_dict['model']() # factory
    crossval_fit = model_dict['crossval_fit']
    # logging.info(model.score(X_test, y_test))

    # run the task:
    X = task_args['X']
    y = task_args['y']
    return (crossval_fit(model, X, y.values.ravel()), model)


  def cross_validate(self):
    check_is_fitted(self)

    run_cfg = self.run_cfg
    tasks = run_cfg['cv_tasks']
    logging.info(f'Cross validating classification models {tasks}')

    X_train, X_test, y_train, y_test = train_test_split(
      self._X, 
      self._y,
      test_size=run_cfg['overfit/test_size']
    )

    task_args = [{
      'X': X_train.copy(deep=True),
      'y': y_train.copy(deep=True),
      'task': t
    } for t in tasks]
    args = [{'estimators': self.estimators,'task_args': a} for a in task_args]

    train_scores = []
    for i, arg in enumerate(args):
      s, m = self.cv_task(arg)
      train_scores.append(s)

    train_scores_mean = pd.DataFrame( np.array(train_scores).T).mean()
    train_scores_mean.index = tasks
    logging.info(train_scores_mean)
    return train_scores_mean

  ##################### Custom functions ########################
  def df_sanitization(self, data_frame):
    copy = data_frame.copy(deep=True)
    npa = copy.to_numpy()
    return pd.DataFrame(
      data=npa, columns=copy.columns.tolist()
    )

  def pca_dim_reduction(self, X, n_comp):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    logging.info(f"\nPC 1 with scaling:\n { pca.components_[0]}")
    return X

  # def normalize(self, X, X_test, method):
  #   scaler_dic = {
  #     'minmax': MinMaxScaler(),
  #     'standard': StandardScaler(),
  #   }
  #   scaler = scaler_dic[method]
  #   scaler = scaler.fit(X)
  #   X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
  #   X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
  #   return X_scaled, X_test_scaled


  def normalize(self, X, method, use_pretrained=False):
    # TODO: save this to pickle
    if use_pretrained and self._scaler_ is not None:
      logging.warn('using pretrained normalizer')
      scaler = self._scaler_
    else:
      scaler = self.scaler_dic[method]()
    scaler = scaler.fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    self._scaler_ = scaler
    return X_scaled


  def autofeat_dim_reduction(self, X,y):
    fsel = FeatureSelector(verbose=1)
    X = fsel.fit_transform(X,y)
    return X


  def simple_fit(self, model, X, y):
    model = model.fit(X, y)
    return model 


  def auto_crossval(self, model, X, y):
    # rkf = RepeatedKFold(n_splits=10, n_repeats=2)
    rkf = StratifiedKFold(n_splits=10)   # better kfold for imbalanced dataset

    scores = cross_val_score(
      model, X, y, cv=rkf, verbose=1,
      scoring='balanced_accuracy'   # For scoring strings, see: https://scikit-learn.org/stable/modules/model_evaluation.html 
    )
    return scores


  def cfg_to_estimators(self, run_cfg):
    # stackedclf_cfg = run_cfg['stackedclf/estimators']

    estimators = {
      'lightgbm': {
        'model': lambda : lgbm.LGBMClassifier(
          boosting_type=run_cfg['models/lightgbm/boosting_type'],
          class_weight=run_cfg['models/lightgbm/class_weight'],
          num_leaves=run_cfg['models/lightgbm/num_leaves'],
          learning_rate=run_cfg['models/lightgbm/learning_rate'],
          n_estimators=run_cfg['models/lightgbm/num_iterations']
        ),
        'fit': self.simple_fit,
        'crossval_fit': lambda m,X,y: self.auto_crossval(m,X,y),
        'validate': lambda m,X,y: m.score(m,X,y)
      },
      'svc': {
        'model': lambda : SVC(
          C=run_cfg['models/svc/C'],
          kernel=run_cfg['models/svc/kernel'],
          class_weight=run_cfg['models/svc/class_weight'],
          gamma=run_cfg['models/svc/gamma'],
          decision_function_shape=run_cfg['models/svc/decision_function_shape']
        ),
        'fit': self.simple_fit,
        'crossval_fit': lambda m,X,y: self.auto_crossval(m,X,y),
        'validate': lambda m,X,y: m.score(m,X,y)
      },
      'gpclf': {
        'model': lambda : GaussianProcessClassifier(
          multi_class=run_cfg['models/gpclf/multi_class']
        ),
        'fit': self.simple_fit,
        'crossval_fit': lambda m,X,y: self.auto_crossval(m,X,y),
        'validate': lambda m,X,y: m.score(m,X,y)
      },
      'perceptron': {
        'model': lambda : Perceptron(
          penalty=run_cfg['models/perceptron/penalty'],
          shuffle=run_cfg['models/perceptron/shuffle'],
          class_weight=run_cfg['models/perceptron/class_weight']
        ),
        'fit': self.simple_fit,
        'crossval_fit': lambda m,X,y: self.auto_crossval(m,X,y),
        'validate': lambda m,X,y: m.score(m,X,y)
      }
      # 'stackedclf'
      #   'model': lambda : StackingClassifier(estimators=stackedclf_cfg, verbose=1),
      #   'fit': self.simple_fit,
      #   'crossval_fit': lambda m,X,y: self.auto_crossval(m,X,y),
      #   'validate': lambda m,X,y: m.score(m,X,y)
    }
    return estimators


  def fill_nan(self, X, strategy):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    return(imputer.fit_transform(X))


  def preprocess_unlabeled(self, run_cfg, X_u):
    X_u[:] = self.fill_nan(X_u, run_cfg['preproc/imputer/strategy'])  # test

    flag_normalize = run_cfg['preproc/normalize/enabled']
    if flag_normalize: 
    #   X, X_u = normalize(X, X_u, run_cfg['preproc/normalize/method'])
      X_u = self.normalize(
        X_u, 
        run_cfg['preproc/normalize/method'], 
        use_pretrained=run_cfg['preproc/normalize/use_pretrained_for_X_u']
      )
    
    return X_u


  def preprocess(self, run_cfg, X, y, X_test):

    # Remove NaN from training and test data
    X[:] = self.fill_nan(X, run_cfg['preproc/imputer/strategy'])  # train
    X_test[:] = self.fill_nan(X_test, run_cfg['preproc/imputer/strategy'])  # test


    # Normalization training data and save to separate variable X_norm
    flag_normalize = run_cfg['preproc/normalize/enabled']
    if flag_normalize:
      X_norm = self.normalize(X, run_cfg['preproc/normalize/method'])

    # Outlier removal from X_norm
    if run_cfg['preproc/outlier/enabled']:
      cont_lim =  run_cfg['preproc/outlier/cont_lim']
      outliers = find_isolation_forest_outlier(X_norm,cont_lim)
      # Remove outliers from original training set
      X = X.drop(index=outliers)
      y = y.drop(index=outliers)

    # Re-normalization of training and test sets without outliers
    flag_normalize = run_cfg['preproc/normalize/enabled']
    if flag_normalize:
      X = self.normalize(X, run_cfg['preproc/normalize/method'])
      X_test =  pd.DataFrame(self._scaler_.transform(X_test), index=X_test.index, columns=X_test.columns)

    # Feature reduction
    rfe_method = run_cfg['preproc/rmf/rfe/method']
    rfe_estimator = run_cfg['preproc/rmf/rfe/estimator']
    rfe_step_size = run_cfg['preproc/rmf/rfe/step_size']
    rfe_min_feat = run_cfg['preproc/rmf/rfe/min_feat']
    rfe_estimator_cfg = run_cfg[f'models/{rfe_estimator}']

    rmf_pipelines = {
      'ffu': lambda X,y: ffu_dim_reduction(run_cfg,X,y),
      'rfe': lambda X,y: rfe_dim_reduction(
        X,y,rfe_method, rfe_estimator,
        estimator_args=rfe_estimator_cfg,
        min_feat = rfe_min_feat,
        step = rfe_step_size
      ),
      'auto' : lambda X,y: self.autofeat_dim_reduction(X,y)
    }
    rmf_pipeline_name = run_cfg['preproc/rmf/pipeline']
    X = rmf_pipelines[rmf_pipeline_name](X,y)

    # # Apply pca (with min max normalization)
    # if run_cfg['preproc/rmf/pca/enabled']:
      # if not flag_normalize: 
      #   logging.error('Unnormalized data as PCA input!')
      # n_comp = run_cfg['preproc/rmf/pca/n_comp']
      # X = self.pca_dim_reduction(X, n_comp)

    return X, y, X_test



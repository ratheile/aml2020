import logging
import os
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .outlier import find_isolation_forest_outlier
from .rmf_rfe import rfe_dim_reduction
from .rmf_pca import pca_dim_reduction, pca_dim_reduction_transform
from .umap import umap_dim_reduction, umap_dim_reduction_transform


from .sampling import balancing

from autofeat import FeatureSelector

from sklearn.metrics import r2_score, balanced_accuracy_score

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

from sklearn.utils import shuffle


from sklearn.linear_model import \
    LinearRegression, \
    Lasso, \
    Ridge, \
    ElasticNet

import hashlib
import logging
import json
import time
from joblib import dump, load

import lightgbm as lgbm
from xgboost import XGBRegressor

class Project2Estimator(BaseEstimator):

  ##################### Estimator API ########################
  def __init__(self, run_cfg, env_cfg, slice_cfg=None, **args):
    self.env_cfg = env_cfg
    self.run_cfg = run_cfg
    self.slice_cfg = slice_cfg
    logging.info(f'Estimator initialized  (addiditonal args added to run_cfg: {args} )')

    for key, value in args.items():
      if key != 'run_cfg' and key != 'env_cfg' and key != 'slice_cfg':
        self.run_cfg[key] = value

    self.scaler_dic = {
      'minmax': lambda: MinMaxScaler(),
      'standard': lambda: StandardScaler(),
    }
    
    self.estimators = self.cfg_to_estimators(run_cfg)

    if slice_cfg is not None:
      self.parameters = list(slice_cfg['run_cfg'].keys())
    else:
      self.parameters = []


  def fit(self, X, y):
    begin_time = time.time()
    
    # Sanitization
    
    # Sometimes indices of dataframes and series 
    # do not match
    X = self.df_sanitization(X) 
    y = self.df_sanitization(y)

    # Check y.values.ravel() because we use this exact 
    # method also to pass our y to other sklearn models
    X_checked, y_checked = check_X_y(X, y.values.ravel())
    X = pd.DataFrame(data=X_checked,columns=X.columns)
    y = pd.DataFrame(data=y_checked,columns=y.columns)


    # Bypass Data Input
    hash_dir = self.env_cfg['datasets/project2/hash_dir']
    skip_preprocessing = False

    dim_red_pipeline = self.run_cfg['preproc/rmf/pipeline']
    load_dim_red =  self.run_cfg[f'preproc/rmf/{dim_red_pipeline}/needs_persistence'] and \
                    self.run_cfg[f'preproc/rmf/enabled']


    df_hash_f = lambda df: hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()
    fn_func = lambda hash_df, hash_cfg, postfix: f'{hash_dir}/{hash_df}_{hash_cfg}_{postfix}'
    
    load_flag = self.run_cfg['persistence/load_from_file']
    save_flag = self.run_cfg['persistence/save_to_file']

    if load_flag or save_flag:
      # hashes
      cfg_hash = hashlib.sha256(json.dumps(self.run_cfg['preproc']).encode()).hexdigest()
      X_hash = df_hash_f(X)
      y_hash = df_hash_f(y)
      # filenames
      X_file = fn_func(X_hash,cfg_hash, 'X.pkl')
      y_file = fn_func(y_hash,cfg_hash, 'y.pkl')
      scaler_file = fn_func(X_hash,cfg_hash, 'scaler.joblib')
      dim_red_file = fn_func(X_hash,cfg_hash, 'dimred.joblib')
      
    self._scaler_ = None
    if load_flag:
      files_present = os.path.isfile(X_file) and \
        os.path.isfile(y_file) and \
        os.path.isfile(scaler_file)
      
      if load_dim_red:
        files_present = files_present and os.path.isfile(dim_red_file)

      if files_present:
        logging.warning(f'Files found to preload config hash {cfg_hash} for dataset.')
        # logging.warning(f'found pickle for X: {X_file}')
        # logging.warning(f'found pickle for y: {y_file}')
        # logging.warning(f'found pickle for normalization model: {scaler_file}')
        X = pd.read_pickle(X_file)
        y = pd.read_pickle(y_file)
        self._scaler_ = load(scaler_file)

        if load_dim_red:
          # logging.warning(f'found pickle for PCA model: {pca_file}')
          self._dim_red_ = load(dim_red_file)

        skip_preprocessing = True


    # Data Preprocessing
    self._preprocessing_skipped_ = skip_preprocessing
    if not skip_preprocessing:
      # preprocess also fits a _scaler_
      X, y = self.preprocess(self.run_cfg, X, y)


    # Store
    if save_flag and not skip_preprocessing:
      X.to_pickle(X_file)
      y.to_pickle(y_file)
      dump(self._scaler_, scaler_file)
      if load_dim_red:
        dump(self._dim_red_, dim_red_file)


    # Regression model fit
    estimator_name = self.run_cfg['fit_model']
    estimator_cfg = self.estimators[estimator_name]
    model = estimator_cfg['model']() # factory
    fitted_model = estimator_cfg['fit'](model, X, y.values.ravel())

    end_time = time.time()
    logging.info(f'Fitting completed in: {end_time - begin_time:.4f} seconds.')

    # Assign local variables
    # also includes (above):
    # - self._preprocessing_skipped_ 
    # - self._scaler_
    # - self._pca_dim_red_
    self._fitted_model_ = fitted_model
    self._X = X
    self._y = y


  def predict(self, X_u):

    check_is_fitted(self)

    X_u = self.df_sanitization(X_u)
    X_u = self.preprocess_unlabeled(self.run_cfg, X_u)

    # Reduce dimensionality of test dataset
    # based on feature selection on training data 
    rmf_pipeline  = self.run_cfg['preproc/rmf/pipeline']
    
    if self.run_cfg['preproc/rmf/enabled']:
      if rmf_pipeline == 'pca':
        logging.info('transforming X_u via pca')
        X_u = pca_dim_reduction_transform(self._dim_red_, X_u)
      elif rmf_pipeline == 'umap':
        logging.info('transforming X_u via umap')
        X_u = umap_dim_reduction_transform(self._dim_red_, X_u)
      elif rmf_pipeline == 'rfe':
        X_u = X_u[self._X.columns]
      else:
        raise ValueError(f'prediction rmf not implemented for {rmf_pipeline}')
    
    y_u = self._fitted_model_.predict(X_u)
    return y_u


  def score(self, X, y=None):
    score_fn = {
      'balanced_accuracy': lambda: balanced_accuracy_score
    }
    return(score_fn[self.run_cfg['scoring']](self.predict(X), y))


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

    logging.info(f'Setting parameters of estimator to {params}')

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


  ##################### Custom functions ########################
  def df_sanitization(self, data_frame):
    copy = data_frame.copy(deep=True)
    npa = copy.to_numpy() # Extract numpy array and regenerate indices
    return pd.DataFrame(
      data=npa, columns=copy.columns.tolist()
    )
  

  def normalize(self, X, method, use_pretrained=False):
    if self._preprocessing_skipped_ and self._scaler_ is None:
      logging.error('Preprocessing skipped: data is probalby normalized with a wrong (empty) scaler!')

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

  def simple_fit(self, model, X, y):  # TODO to ask: do we need this?
    model = model.fit(X, y)
    return model 

  def cfg_to_estimators(self, run_cfg):
    # stackedclf_cfg = run_cfg['stackedclf/estimators']

    estimators = {
      'lightgbm': {
        'model': lambda : lgbm.LGBMClassifier(**run_cfg['models/lightgbm']),
        'fit': self.simple_fit,
        'validate': lambda m,X,y: m.score(m,X,y)
      },
      'svc': {
        'model': lambda : SVC(**run_cfg['models/svc']),
        'fit': self.simple_fit,
        'validate': lambda m,X,y: m.score(m,X,y)
      },
      'gpclf': {
        'model': lambda : GaussianProcessClassifier(
          multi_class=run_cfg['models/gpclf/multi_class']
        ),
        'fit': self.simple_fit,
        'validate': lambda m,X,y: m.score(m,X,y)
      },
      'perceptron': {
        'model': lambda : Perceptron(
          penalty=run_cfg['models/perceptron/penalty'],
          shuffle=run_cfg['models/perceptron/shuffle'],
          class_weight=run_cfg['models/perceptron/class_weight']
        ),
        'fit': self.simple_fit,
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


  def preprocess(self, run_cfg, X, y):

    X, y = shuffle(X, y) # https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html

    # Remove NaN from training and test data
    fill_nan_configured = lambda X: self.fill_nan(X, run_cfg['preproc/imputer/strategy'])  # train
  
    X[:] = fill_nan_configured(X)  # train

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
    if flag_normalize:
      X = self.normalize(X, run_cfg['preproc/normalize/method'])

    if run_cfg['preproc/balancing/enabled']:
      method = run_cfg['preproc/balancing/method']

      if f'preproc/balancing/models/{method}' in run_cfg:
        method_arg_dict = run_cfg[f'preproc/balancing/models/{method}']
        logging.info(f'balancing ({method}) args: {method_arg_dict}')
        X,y = balancing(X,y, method_arg_dict, method)
      else:
        logging.warning(f'Balancing with an unconfigured method {method}')
        X,y = balancing(X,y, method)


    # Feature reduction
    if run_cfg['preproc/rmf/enabled']:
      rfe_method = run_cfg['preproc/rmf/rfe/method']
      rfe_step_size = run_cfg['preproc/rmf/rfe/step_size']
      rfe_estimator = run_cfg['preproc/rmf/rfe/estimator']
      rfe_min_feat = run_cfg['preproc/rmf/rfe/min_feat']
      rfe_estimator_args = run_cfg[f'preproc/rmf/rfe/models/{rfe_estimator}']
      
      pca_method = run_cfg['preproc/rmf/pca/method']
      pca_estimator_args = run_cfg['preproc/rmf/pca/model']

      umap_args = run_cfg['preproc/rmf/umap/model']
  
      rmf_pipelines = {
        'rfe': lambda X,y: rfe_dim_reduction(  # TODO to ask: do you know if we can just pass in the dictionary of options instead of creating a ton of variables for this?
          X,y,rfe_method, rfe_estimator,
          estimator_args=rfe_estimator_args,
          min_feat = rfe_min_feat,
          step = rfe_step_size
        ),
        'umap': lambda X,y: umap_dim_reduction(X,y, umap_args=umap_args),
        'pca': lambda X,y: pca_dim_reduction(
            X,y,
            pca_method=pca_method,
            pca_args=pca_estimator_args
          ),
        'auto' : lambda X,y: self.autofeat_dim_reduction(X,y)
      }
      rmf_pipeline_name = run_cfg['preproc/rmf/pipeline']
  
      if rmf_pipeline_name == 'rfe':
        logging.info(f'RFE args: {rfe_estimator_args}')
        X = rmf_pipelines[rmf_pipeline_name](X,y)
      elif rmf_pipeline_name == 'pca':
        logging.info(f'PCA args: {rfe_estimator_args}')
        X, pca = rmf_pipelines[rmf_pipeline_name](X,y)
        self._dim_red_ = pca
      elif rmf_pipeline_name == 'umap':
        logging.info(f'UMAP args: {rfe_estimator_args}')
        X, umap = rmf_pipelines[rmf_pipeline_name](X,y)
        self._dim_red_ = umap 
      else:
        raise ValueError(f'rmf not implemented for {rmf_pipeline_name}')
      
    # at this point we also (optionally) created:
    # self._scaler_
    # self._pca_dim_red_
    return X, y 
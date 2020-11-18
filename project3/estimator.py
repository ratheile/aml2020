import hashlib
import json
import logging
import multiprocessing
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load

# ML libraries
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# ECG libraries
import biosppy


class Project3Estimator(BaseEstimator):

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
    
    # Bypass Data Input
    hash_dir = self.env_cfg['datasets/project3/hash_dir']
    skip_preprocessing = False

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

      if files_present:
        logging.warning(f'Files found to preload config hash {cfg_hash} for dataset.')
        # logging.warning(f'found pickle for X: {X_file}')
        # logging.warning(f'found pickle for y: {y_file}')
        # logging.warning(f'found pickle for normalization model: {scaler_file}')
        X = pd.read_pickle(X_file)
        y = pd.read_pickle(y_file)
        self._scaler_ = load(scaler_file)

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

    # Shuffle
    X, y = shuffle(X, y) # https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html

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
    X_u = self.preprocess(self.run_cfg, X_u)
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

  def preprocess(self, run_cfg, X, y):
    
    new_feat = [
      'mean_HR',
      'std_HR',
      'amplitude_P_wave',
      'std_S_amplitude',
      'mean_QRS_duration',
      'std_QRS_duration'
    ]

    X_new = pd.DataFrame(columns=new_feat)

    for i in range(X.shape[0]):
      X_new.iloc[i,:] = preprocess_time_series(X.iloc[i,:])

    return X_new, y 

  def preprocess_time_series(self, run_cfg, x): #TODO: main function to complete!

    # 0. Remove NaNs
    x = x.dropna().to_numpy()

    # 0. Drop first part of signal: noisy TODO: entweder ganze nehmen oder Threshold definieren

    # 1. Detection and exclusion of class 3 from training set (TODO Raffi)

    # 2. Detection of flipped signals and flipping (TODO Raffi + Inês)

    # 3. Filtering (getting isoelectric line and smoothing)
    ecg_info = biosppy.signals.ecg.ecg(
      signal=ecg, 
      sampling_rate=sampling_rate, 
      show=True
      )
    ecg_filtered = ecg_info['filtered'] # Extract filtered data
    
    x.loc[n,'mean_HR']=np.mean(ecg_info['heart_rate'])
    x.loc[n,'std_HR']=np.std(ecg_info['heart_rate'])

    # 4. Waveform detection
    #   4.1 R-peaks and HR: 
    #         - mean_HR (class 1)
    #         - std_HR (class 1)
    #   4.2 P, QRS and T: TODO (Francesco)
    #         - number of P waves, amplitude_P_wave (class 1)
    #         - mean_S_amplitude, ?? std_S_amplitude (class 2)
    #         - mean_QRS_duration, (class 2)
    #         - std_QRS_duration (class 2)
    return x

  def simple_fit(self, model, X, y):  # TODO to ask: do we need this?
    model = model.fit(X, y)
    return model 

  def cfg_to_estimators(self, run_cfg):

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
        ),
        'fit': self.simple_fit,
        'validate': lambda m,X,y: m.score(m,X,y)
      }
    }
    return estimators
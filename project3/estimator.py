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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import f1_score

# ECG libraries
import biosppy
from .feature_extraction import \
  split_classes, \
  calc_peak_summary, \
  extract_features

import lightgbm as lgbm


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
    
    preproc_enabled = self.run_cfg['preproc/enabled']
    if preproc_enabled:
      logging.warning('preprocessing enabled for X in fit()')
      logging.warning('preprocessing enabled: this mode should only be used in run')
      # Bypass Data Input
      hash_dir = self.env_cfg['datasets/project3/hash_dir']
      skip_preprocessing = False

      df_hash_f = lambda df: hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()
      fn_func = lambda hash_df, hash_cfg, postfix: f'{hash_dir}/{hash_df}_{hash_cfg}_{postfix}'
      
      load_flag = self.run_cfg['persistence/load_from_file']
      save_flag = self.run_cfg['persistence/save_to_file']
      save_plot_data = self.run_cfg['persistence/save_plot_data']

      if load_flag or save_flag:
        # hashes
        cfg_hash = hashlib.sha256(json.dumps(self.run_cfg['preproc']).encode()).hexdigest()
        X_hash = df_hash_f(X)
        y_hash = df_hash_f(y)
        # filenames
        X_file = fn_func(X_hash, cfg_hash, 'X.pkl')
        y_file = fn_func(y_hash, cfg_hash, 'y.pkl')
        scaler_file = fn_func(X_hash, cfg_hash, 'scaler.joblib')
        dim_red_file = fn_func(X_hash, cfg_hash, 'dimred.joblib')
        
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
        X, y, X_plot_data, no_nan_mask = self.preprocess(X, y)

        X = X[no_nan_mask]
        y = y[no_nan_mask]

        # Address NaNs TODO: still necessary
        #TODO (check why this happens): 
        # With median, we sometimes get negative durations for QRS_t_mean
        X[:] = self.fill_nan(run_cfg=self.run_cfg, X=X)

      # Store
      if save_flag and not skip_preprocessing:
        X.to_pickle(X_file)
        y.to_pickle(y_file)
        dump(self._scaler_, scaler_file)

        if save_plot_data: 
          plot_data_file = fn_func(X_hash, cfg_hash, 'plotData.joblib')
          dump(X_plot_data, plot_data_file) 
    else:
      logging.warning('preprocessing disabled for X in fit()')
      logging.warning('preprocessing disabled: this mode should only be used in GridSearchCV')
    

    # delete plot data to free memory
    X_plot_data = None

    # drop features we don't need
    drop_features=self.run_cfg['drop_features/enabled']
    dropped_features_list=self.run_cfg['drop_features/dropped_features']
    if drop_features:
      X = X.drop(columns=dropped_features_list)

    #normalize
    flag_normalize = self.run_cfg['preproc/normalize/enabled']
    if flag_normalize:
      X = self.normalize(X, self.run_cfg['preproc/normalize/method'])

    # Shuffle after preprocessing and before training
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

    # X_u = self.df_sanitization(X_u) # TODO: check if still necessary

    save_x_u = self.run_cfg['persistence/save_x_unlabeled']
    load_x_u = self.run_cfg['persistence/load_x_unlabeled']

    preproc_enabled = self.run_cfg['preproc/enabled']
    if preproc_enabled:
      X_u, _, X_u_plotData, no_nan_mask = self.preprocess(X_u)
      
      logging.warning(f'Length after preprocess: X_u, no_nan_mask = {X_u.shape[0]}, {len(no_nan_mask)}.')
    
      X_u = X_u[no_nan_mask] 

      # Address NaNs TODO: still necessary
      # TODO (check why this happens): 
      # With median, we sometimes get negative durations for QRS_t_mean
      X_u[:] = self.fill_nan(run_cfg=self.run_cfg, X=X_u)

      if save_x_u:
        # hashes
        hash_dir = self.env_cfg['datasets/project3/hash_dir']
        df_hash_f = lambda df: hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()
        fn_func = lambda hash_df, hash_cfg, postfix: f'{hash_dir}/{hash_df}_{hash_cfg}_{postfix}'
        cfg_hash = hashlib.sha256(json.dumps(self.run_cfg['preproc']).encode()).hexdigest()
        X_u_hash = df_hash_f(X_u)
        # filenames
        X_u_file = fn_func(X_u_hash, cfg_hash, 'X_u.joblib')
        dump([no_nan_mask, X_u], X_u_file)

    else:
      logging.warning('preprocessing disabled for unlabled X in predict()')
      # WARNING HACK X_U is preprocessed input
      if load_x_u:
        temp = X_u
        X_u = temp[1]
        no_nan_mask = temp[0]
    
    drop_features=self.run_cfg['drop_features/enabled']
    dropped_features_list=self.run_cfg['drop_features/dropped_features']
    if drop_features:
      X_u = X_u.drop(columns=dropped_features_list)

    #normalize
    flag_normalize = self.run_cfg['preproc/normalize/enabled']
    if flag_normalize: 
    #   X, X_u = normalize(X, X_u, run_cfg['preproc/normalize/method'])
      X_u = self.normalize(
        X_u, 
        self.run_cfg['preproc/normalize/method'], 
        use_pretrained=self.run_cfg['preproc/normalize/use_pretrained_for_X_u']
      )
    
    # we call predict only on a valid subset of X (not nan)
    y_u = self._fitted_model_.predict(X_u)

    # if we have done preprocessing, then we might have samples that failed 
    # (crashed) the preprocessor. this is not the case if we only use
    # successfully preprocessed data
    if preproc_enabled or load_x_u:
      y_total = np.zeros(len(no_nan_mask))
      y_total[no_nan_mask] = y_u
      y_total[np.logical_not(no_nan_mask)] = 0
      # TODO take care of last few unlabeled masked nan samples
    else:
      y_total = y_u

    return y_total 

  def score(self, X, y=None):
    score_fn = {
      # scoring
      'f1_micro': lambda: f1_score
    }
    return(score_fn[self.run_cfg['scoring']](self.predict(X), y, average='micro'))

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

  def fill_nan(self, run_cfg, X):
    imputers ={
      'simple': lambda: SimpleImputer(missing_values=np.nan, strategy=run_cfg['preproc/imputer/strategy']),
      'knn': lambda: KNNImputer(
        missing_values=np.nan,
        n_neighbors=run_cfg['preproc/imputer/n_neighbors'],
        weights=run_cfg['preproc/imputer/weights']
        )
    }
    imputer = imputers[run_cfg['preproc/imputer/type']]
    return(imputer().fit_transform(X))

  def normalize(self, X, method, use_pretrained=False, preproc_enabled=False):
    if preproc_enabled and self._preprocessing_skipped_ and self._scaler_ is None:
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

  def preprocess(self, X, y=None):

    # TODO: address the following open problems
    # 1. Detection and exclusion of class 3 from training set (TODO Raffi)
    # 2. Detection of flipped signals and flipping (Raffi + Inês) (DONE): TODO: needs testing
    #     - Proposal from Inês: Measure amplitude of signal at Q peaks and S peaks. Pseudocode: 
    #       If abs(mean(amplitude_Q_peak)) > abs(mean(amplitude_R_peak)) OR abs(mean(amplitude_S_peak)) > abs(mean(amplitude_R_peak))
    #         time_series = -time_series
    #       (I am not suggesting using the mean amplitude of P and T waves and checking whether they are negative because, 
    #       with flipped signals, the detection of these waves is not great, as can be seen from Raffi's visualizer. On the 
    #       hand, Q R ans S waves are still quite detected, as they are very prominent)
    # 3. NaN? When biosppy or neurokit crashen, QRS mean or SD when offsets and onsets not the same are.
    #      - Diagnostics using Raffi's viewer as first step.
    #      - Exclude observations?
    #      - (DONE) FillNaN from Project 1
    #      - (DONE) KNN Imputer: TODO: needs testing
    #      - (DONE) KNN imputation would be better done on class by class basis. Not feasible for predict
    # 4. Handle emply slices: NaN. (TODO Franc) this comes from neurokit directly (confirm) and we have a lot of them... is it related to crashes?
    # 5. Handle missing samples: (TODO Franc) check whether missing samples stem from crashes and why we have so many of them
    # 6. Check out extracted features:  (TODO Franc) check whether feature info allows classification
    # 7. predict y indexing: TODO keep original index (although we might loose samples)
    # 8. cv run: TODO needs testing and maybe fixing

    # Define features to extract


    X_new, y_new, X_new_plotData, no_nan_mask = extract_features(
                            run_cfg=self.run_cfg,
                            env_cfg=self.env_cfg,
                            df=X,
                            y=y,
                            verbose=self.run_cfg['preproc/verbose']
                            )

    return X_new, y_new, X_new_plotData, no_nan_mask

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
      'rfc': {
        'model': lambda : RandomForestClassifier(**run_cfg['models/rfc']),
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
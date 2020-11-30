import logging
import os
import multiprocessing
import joblib
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .estimator import Project3Estimator

from sklearn.model_selection import \
    RepeatedKFold, \
    GridSearchCV, \
    cross_val_score, \
    train_test_split, \
    StratifiedKFold

def gridsearch(run_cfg, env_cfg, slice_cfg): 
  '''
  The idea is to run gridsearch and see which model is the best by looking at the *_grid_results.csv
  Once you can define which model is the winner, adapt base_cfg.yml so you have this set of parameters
  and run 'python main.py --cfg project2/base_cfg.yml' in your terminal to get the predictions from that model.
  ''' 

  # Load training dataset from joblib files 
  X, y, _ = load_preprocessed_pickle(env_cfg)

  # Remove index column
  # y = y.iloc[:,1:] # not present in pickles
  # X = X.iloc[:,1:]
  logging.info('Training dataset imported')

  p3e = Project3Estimator(run_cfg, env_cfg, slice_cfg)
  param_grid = slice_cfg['run_cfg']

  clf = GridSearchCV(
      estimator=p3e,
      param_grid=param_grid,
      scoring=run_cfg['scoring'],
      # Number of jobs to run in parallel
      n_jobs=env_cfg['n_jobs'],
      verbose=env_cfg['verbose']
    )

  clf.fit(X, y)

  results_df = pd.concat([
    pd.DataFrame(clf.cv_results_["params"]),
    pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"]),
    pd.DataFrame(clf.cv_results_['std_test_score'], columns=['SD on test folds']) # info from: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
  ],axis=1)

  if not os.path.exists('predictions'):
    os.makedirs('predictions')

  results_df.to_csv(
      f'predictions/{slice_cfg["experiment_name"]}_grid_results.csv', 
      index=False) # TODO for Euler: reporting needs to work on Euler as well

  logging.info('GridSearchCV complete')

def convert_data(run_cfg, env_cfg): 
  begin_time = time.time()

  # Load training dataset from csv
  datapath = env_cfg['datasets/project3/path']
  logging.info('CSV loading started ...')
  X = pd.read_csv(f'{datapath}/X_train.csv')
  y = pd.read_csv(f'{datapath}/y_train.csv')
  X_u = pd.read_csv(f'{datapath}/X_test.csv') # unlabeled

  joblib.dump(X, f'{datapath}/X_train.joblib')
  joblib.dump(y, f'{datapath}/y_train.joblib')
  joblib.dump(X_u, f'{datapath}/X_test.joblib')

  end_time = time.time()
  logging.info(f'Data conversion done in {end_time - begin_time}')




def load_preprocessed_pickle(env_cfg):
  begin_time = time.time()
  logging.info('Pickle loading started ...')
  datapath = env_cfg['datasets/project3/path']
  X = pd.read_pickle(f'{datapath}/X_train.pkl')
  y = pd.read_pickle(f'{datapath}/y_train.pkl')
  X_u = joblib.load(f'{datapath}/X_test.joblib')

  end_time = time.time()
  logging.info(f'Pickle loading done in {end_time - begin_time}')
  return X, y, X_u


def load_data(env_cfg):
  begin_time = time.time()
  logging.info('Joblib loading started ...')
  datapath = env_cfg['datasets/project3/path']
  X = joblib.load(f'{datapath}/X_train.joblib')
  y = joblib.load(f'{datapath}/y_train.joblib')
  X_u = joblib.load(f'{datapath}/X_test.joblib')
  end_time = time.time()
  logging.info(f'Joblib loading done in {end_time - begin_time}')
  return X, y, X_u


def run(run_cfg, env_cfg):
  '''
  Fits a single model and returns predictions. To be used for prototyping or after
  performing GridSearch.
  '''

  # Load training dataset from joblib files 
  X, y, X_u = load_data(env_cfg)
  # X, y, X_u = load_preprocessed_pickle(env_cfg)
  
  # Remove ID column
  y = y.iloc[:, 1:]
  X = X.iloc[:, 1:]
  X_u = X_u.iloc[:, 1:]
  logging.info('Training dataset imported')

  p3e = Project3Estimator(run_cfg, env_cfg)
  p3e.fit(X, y)  # Needed to do preprocessing. Under sklearn guidelines this is what you should do.
  y_u = p3e.predict(X_u)
  
  if len(y_u.shape) > 1:
    yuf = y_u.flatten()
    y_u = yuf
  y_u_df = pd.DataFrame({
    'id': np.arange(0, len(y_u)).astype(float),
    'y': y_u
  })

  if not os.path.exists('predictions'):
    os.makedirs('predictions')

  estimator_name = run_cfg['fit_model']
  y_u_df.to_csv(
    f'predictions/{estimator_name}_y.csv', 
    index=False)

  logging.info(f'{estimator_name} | Predictions for submission on AML platform saved.')


def cross_validate(run_cfg, env_cfg):
  '''
  Cross validates a single model.
  '''

  # Load training dataset from joblib files 
  # X, y, X_u = load_data(env_cfg)
  X, y, _ = load_preprocessed_pickle(env_cfg)

  # Remove index column
  # y = y.iloc[:,1:]# not present in pickles
  # X = X.iloc[:,1:]
  logging.info('Training dataset imported')

  p3e = Project3Estimator(run_cfg, env_cfg)
  logging.info(f'Cross validation started')

  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=run_cfg['cross_validation/test_set_size']
  )
  rkf = StratifiedKFold(
    n_splits=run_cfg['cross_validation/n_splits']
  )   # better kfold for imbalanced dataset

  scores = cross_val_score(
    p3e, X, y, 
    cv=rkf,
    scoring=run_cfg['scoring'],
    n_jobs=env_cfg['n_jobs'],
    verbose=env_cfg['verbose']
  )

  columns = [
    *scores, # flatten
    np.mean(scores),
    np.std(scores)
  ]

  titles = [
    *[f"cv{i}" for i in range(len(scores))],
    'mean',
    'std'
  ]

  logging.info(f'CV results: {columns}')
  # default orientation is rows -> transpose
  results = pd.DataFrame(columns, index=titles).T

  if 'model_path' in run_cfg:
    model_path = run_cfg['model_path']
    dir_name, file_name = os.path.split(model_path)

    if not os.path.exists(dir_name):
      os.makedirs(dir_name)

    results.to_csv( f'{model_path}.csv', index=False) 

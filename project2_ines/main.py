import logging
import os
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .estimator import Project1Estimator
from sklearn.model_selection import GridSearchCV

def gridsearch(run_cfg, env_cfg, slice_cfg):  # Load training dataset from csv
  datapath = env_cfg['datasets/project1/path']
  X = pd.read_csv(f'{datapath}/X_train.csv')
  y = pd.read_csv(f'{datapath}/y_train.csv')
  X_u = pd.read_csv(f'{datapath}/X_test.csv') # unlabeled
  
  # Remove index column
  y = y.iloc[:,1:]
  X = X.iloc[:,1:]
  logging.info('Training dataset imported')

  p1e = Project1Estimator(run_cfg, env_cfg, slice_cfg)
  param_grid = slice_cfg['run_cfg']

  clf = GridSearchCV(
      estimator=p1e,
      param_grid=param_grid,
      n_jobs=1
    )

  clf.fit(X, y)

  results_df = pd.concat([
    pd.DataFrame(clf.cv_results_["params"]),
    pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"])
  ],axis=1)

  if not os.path.exists('predictions'):
    os.makedirs('predictions')


  results_df.to_csv(
      f'predictions/{slice_cfg["experiment_name"]}_grid_results.csv', 
      index=False)

  logging.info('GridSearchCV complete')

def run(run_cfg, env_cfg):

  # Load training dataset from csv
  datapath = env_cfg['datasets/project1/path']
  X = pd.read_csv(f'{datapath}/X_train.csv')
  y = pd.read_csv(f'{datapath}/y_train.csv')
  X_u = pd.read_csv(f'{datapath}/X_test.csv') # unlabeled
  
  # Remove index column
  y = y.iloc[:,1:]
  X = X.iloc[:,1:]
  logging.info('Training dataset imported')

  p1e = Project1Estimator(run_cfg, env_cfg)
  p1e.fit(X,y)
  scores = p1e.cross_validate()
  y_u = p1e.predict(X_u)
  
  if len(y_u.shape) > 1:
    yuf = y_u.flatten()
    y_u = yuf
  y_u_df =  pd.DataFrame({
    'id': np.arange(0,len(y_u)).astype(float),
    'y': y_u
  })

  if not os.path.exists('predictions'):
    os.makedirs('predictions')

  estimator_name = run_cfg['fit_model']
  y_u_df.to_csv(
    f'predictions/{estimator_name}_y.csv', 
    index=False)
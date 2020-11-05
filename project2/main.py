import logging
import os
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .estimator import Project2Estimator
from sklearn.model_selection import GridSearchCV

def gridsearch(run_cfg, env_cfg, slice_cfg): 
  '''
  The idea is to run gridsearch and see which model is the best by looking at the *_grid_results.csv
  Once you can define which model is the winner, adapt base_cfg.yml so you have this set of parameters
  and run 'python main.py --cfg project2/base_cfg.yml' in your terminal to get the predictions from that model.
  ''' 
  
  # Load training dataset from csv
  datapath = env_cfg['datasets/project2/path']
  X = pd.read_csv(f'{datapath}/X_train.csv')
  y = pd.read_csv(f'{datapath}/y_train.csv')
  X_u = pd.read_csv(f'{datapath}/X_test.csv') # unlabeled
  
  # Remove index column
  y = y.iloc[:,1:]
  X = X.iloc[:,1:]
  logging.info('Training dataset imported')

  p2e = Project2Estimator(run_cfg, env_cfg, slice_cfg)
  param_grid = slice_cfg['run_cfg']

  clf = GridSearchCV(
      estimator=p2e,
      param_grid=param_grid,
      scoring=run_cfg['scoring'],
      n_jobs=1  # Number of jobs to run in parallel
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

def run(run_cfg, env_cfg):
  '''
  Fits a single model and returns predictions. To be used for prototyping or after
  performing GridSearch.
  '''

  # Load training dataset from csv
  datapath = env_cfg['datasets/project2/path']
  X = pd.read_csv(f'{datapath}/X_train.csv')
  y = pd.read_csv(f'{datapath}/y_train.csv')
  X_u = pd.read_csv(f'{datapath}/X_test.csv') # unlabeled
  
  # Remove index column
  y = y.iloc[:,1:]
  X = X.iloc[:,1:]
  X_u = X_u.iloc[:,1:]
  logging.info('Training dataset imported')

  p2e = Project2Estimator(run_cfg, env_cfg)
  p2e.fit(X,y)  # Needed to do preprocessing. Under sklearn guidelines this is what you should do.
  scores = p2e.cross_validate()  # TODO: make this work on Euler and save data
  y_u = p2e.predict(X_u)
  
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

  logging.info(f'{estimator_name} | Predictions for submission on AML platform saved.')
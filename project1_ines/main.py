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
    logging.warn("outlier detection ... done")

  #Subtask 2: Feature selection
  if run_cfg['preprocessing/dim_red']:
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
  result_f = options_dict[run_cfg['task/variant']]
  logging.info(result_f(run_cfg['task']['cfg'])) 
  logging.info(result_f(run_cfg['task/cfg'])) 
  
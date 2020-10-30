# Prototyping script
# Important reading: https://scikit-learn.org/stable/modules/multiclass.html

# List for Gridsearch and CV:
# Outlier detection: cont_lim
# Classifiers: basically everything, including ovr, ovo and multinomial

# Classifiers ordered by performance thus far:
# 1. SVC with rbf kernel (BMAC = 0.7311353721635423)
# 2. Multinomial logistic regression (BMAC = 0.6989916695787445)
# 3. lightggm (BMAC = 0.6334097988599686)
# 4. The rest (BMAC = 0.56)

#%% Imports
import pandas as pd
import numpy as np
import logging

import os
repopath = '/Users/inespereira/Documents/Github/aml2020'
os.chdir(repopath)

from project1.outlier import find_isolation_forest_outlier
from project1.rmf_rfe import rfe_dim_reduction

from modules import ConfigLoader
run_cfg_path = 'project2_ines/base_cfg.yml'
run_cfg = ConfigLoader().from_file(run_cfg_path)

from sklearn.model_selection import train_test_split

# Support cost-sensitive classification
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

# Other classifiers
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import balanced_accuracy_score

#%% Load training dataset from csv
X = pd.read_csv(f'{repopath}/project2_ines/X_train.csv')
X = X.iloc[:,1:]
y = pd.read_csv(f'{repopath}/project2_ines/y_train.csv')
y = y.iloc[:,1:]
X_test = pd.read_csv(f'{repopath}/project2_ines/X_test.csv')
logging.info('I have imported your training dataset! :D')
print(f'Shape of training set is {X.shape}')
print(X)

#%% Check for NaNs
X.isnull().values.any() # No NaNs!

#%% Remove outliers?
X_inliers, y_inliers = find_isolation_forest_outlier(X,y,cont_lim=0.05)
print(f"{len(X)-len(X_inliers)} outliers removed.")

X = X_inliers
y = y_inliers
#%% RFECV ?
rfe_estimator = 'lightgbm'
rfe_estimator_cfg = run_cfg[f'models/{rfe_estimator}']
X_red = rfe_dim_reduction(X,y,method='rfecv',estimator=rfe_estimator,estimator_args=rfe_estimator_cfg, min_feat=20, step=50, verbose=1)

#%%
X_red.describe()

#%% Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X_red, y, test_size=0.1)

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

#%% Look at train and test sets
# X_train

#%% Define the different classifiers you want to look at

classifiers={
  'gpclf': lambda: GaussianProcessClassifier(multi_class='one_vs_one'), # OvO and OvR
  'perceptron': lambda: Perceptron(
    penalty='elasticnet',
    shuffle=True,
    verbose=1,
    # The “balanced” mode uses the values of y to automatically adjust 
    # weights inversely proportional to class frequencies in the input data 
    # as n_samples / (n_classes * np.bincount(y))
    class_weight='balanced'),
  'linearSVM': lambda: LinearSVC(
    penalty='l2',
    loss='squared_hinge',
    C=1.0, # Regularization parameter
    multi_class='ovr',
    class_weight='balanced',
    verbose=1,
    max_iter=1000),
  # Difference between SVM and SVC? https://stackoverflow.com/questions/27912872/what-is-the-difference-between-svc-and-svm-in-scikit-learn
  'SVC': lambda: SVC(
    C=1.0, 
    kernel='rbf', 
    degree=3, # ignored if kernel not poly
    gamma='scale', 
    coef0=0.0, # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    shrinking=True, # What it this?
    probability=False, 
    tol=0.001, 
    class_weight='balanced', 
    verbose=True, 
    max_iter=-1, # no limit
    decision_function_shape='ovo', #ovo or ovr
    break_ties=False, 
    random_state=None),
  'logreg': lambda: LogisticRegression(
    penalty='l2', 
    dual=False, # Dual or primal formulation?
    tol=0.0001, # Tolerance for stopping criteria.
    C=1.0, # Inverse of regularization strength; must be a positive float. 
    # Like in support vector machines, smaller values specify stronger regularization.
    fit_intercept=True, 
    intercept_scaling=1, 
    class_weight='balanced', 
    random_state=None, 
    solver='lbfgs', # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss
    max_iter=100, 
    multi_class='multinomial', # If the option chosen is ‘ovr’, then a binary 
    # problem is fit for each label. For ‘multinomial’ the loss minimised is 
    # the multinomial loss fit across the entire probability distribution, 
    # even when the data is binary. 
    verbose=1, 
    warm_start=False, 
    n_jobs=None, 
    l1_ratio=None),
  'lightgbm': lambda: LGBMClassifier(
    boosting_type='dart', 
    num_leaves=31, 
    max_depth=- 1, 
    learning_rate=0.1, 
    n_estimators=1000, 
    subsample_for_bin=200000, 
    objective=None, 
    class_weight='balanced', 
    min_split_gain=0.0, 
    min_child_weight=0.001, 
    min_child_samples=20, 
    subsample=1.0, 
    subsample_freq=0, 
    colsample_bytree=1.0, 
    reg_alpha=0.0, 
    reg_lambda=0.0, 
    random_state=None, 
    n_jobs=- 1, 
    silent=True, 
    importance_type='split')
}

#%% Choose one classifier and train
clf = classifiers['lightgbm']()
clf = clf.fit(X_train,y_train)

# %%
y_pred = clf.predict(X_test)
# %%
BMAC = balanced_accuracy_score(y_test, y_pred)
print(BMAC)
# %%
# X_red.to_csv('best_preprocessed_X.csv')

# %%

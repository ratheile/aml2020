# Prototyping script
# Important reading: https://scikit-learn.org/stable/modules/multiclass.html

# List for Gridsearch and CV:
# Outlier detection: cont_lim
# Classifiers: basically everything
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
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# Other classifiers
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import balanced_accuracy_score

#%% Load training dataset from csv
X = pd.read_csv(f'{repopath}/project2_ines/X_train.csv')
X = X.iloc[:,1:]
y = pd.read_csv(f'{repopath}/project2_ines/y_train.csv')
y = y.iloc[:,1:]
X_u = pd.read_csv(f'{repopath}/project2_ines/X_test.csv')
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
    random_state=None)
}

#%% Choose one classifier and train
clf = classifiers['SVC']()
clf = clf.fit(X_train,y_train)

# %%
y_pred = clf.predict(X_test)
# %%
BMAC = balanced_accuracy_score(y_test, y_pred)
print(BMAC)
# %% Try on real test data
X_u = X_test.iloc[:,1:]
X_u = pd.DataFrame(X_u, columns=X_red.columns)

# %%
y_pred = clf.predict(X_u)

# %%
submissions =  pd.DataFrame({
  'id': np.arange(0,len(y_pred)).astype(float),
  'y': y_pred
})
submissions.to_csv('project2_ines/prototyping_svc/rbf-kernel-svc.csv', index=False)

print(submissions)
# %%

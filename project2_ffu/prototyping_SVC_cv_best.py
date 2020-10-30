# Prototyping script
# Important reading: https://scikit-learnorg/stable/modules/multiclass.html

# List for Gridsearch and CV:
# Outlier detection: cont_lim
# Classifiers: basically everything
#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

import os
repopath = '/Users/francescofusaro/Documents/GitHub/aml2020'
os.chdir(repopath)

#from project1.outlier import find_isolation_forest_outlier
#from project1.rmf_rfe import rfe_dim_reduction

#from modules import ConfigLoader
#run_cfg_path = 'project2_ines/base_cfg.yml'
#run_cfg = ConfigLoader().from_file(run_cfg_path)

from sklearn.model_selection import train_test_split

# Support cost-sensitive classification
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Other classifiers
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix

#%% Load training dataset from csv
X = pd.read_csv(f'{repopath}/task2/X_train.csv')
X = X.iloc[:,1:]
y = pd.read_csv(f'{repopath}/task2/y_train.csv')
y = y.iloc[:,1:]
X_test = pd.read_csv(f'{repopath}/task2/X_test.csv')
logging.info('I have imported your training dataset! :D')
print(f'Shape of training set is {X.shape}')
print(X)

#%% Check for NaNs
X.isnull().values.any() # No NaNs!

#%% Remove outliers?
# X_inliers, y_inliers = find_isolation_forest_outlier(X,y,cont_lim=0.05)
# print(f"{len(X)-len(X_inliers)} outliers removed.")

# X = X_inliers
# y = y_inliers
#%% RFECV ?
# rfe_estimator = 'lightgbm'
# rfe_estimator_cfg = run_cfg[f'models/{rfe_estimator}']
# X_red = rfe_dim_reduction(X,y,method='rfecv',estimator=rfe_estimator,estimator_args=rfe_estimator_cfg, min_feat=20, step=50, verbose=1)
X_red = X
#%%
X_red.describe()

#%% Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X_red, y, test_size=0.1)

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

#%% Look at train and test sets class weights
y_train_ar = y_train.to_numpy().flatten()
n_samples=y_train_ar.shape[0]
bins = np.bincount(y_train_ar)
n_classes=bins.shape[0]
weights = n_samples / (n_classes * bins)


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
    #class_weight='balanced',
    class_weight={0: weights[0], 1: weights[1]*1.2, 2: weights[2]},
    verbose=True, 
    max_iter=-1, # no limit
    decision_function_shape='ovo', #ovo or ovr
    break_ties=False, 
    random_state=None),
  'xgboost': lambda: XGBClassifier(
    learning_rate=0.05, # default 0.3
    min_child_weight=5, # default 1
    max_depth=12, # default 6
    n_estimators=1000, # default 100
    scale_pos_weight=1), #total_negative_examples / total_positive_examples
  'lightgbm': lambda: LGBMClassifier(
    num_leaves=127, # default 31
    learning_rate=0.1, # default 0.1
    num_iterations=1000, # default 100
    boosting_type="dart", # default gbdt [dart, goss, rf]
    multiclass_ova="True",
    num_class=3,
    class_weight="balanced", #Use this parameter only for multi-class classification task : n_samples / (n_classes * np.bincount(y))
    #scale_pos_weight=67, #used only in binary and multiclassova applications
    #is_unbalance="True", #used only in binary and multiclassova applications. Note: this parameter cannot be used at the same time with is_unbalance, choose only one of them
    )
}

#%% Choose one classifier and train (using cross validation)
clf = classifiers['SVC']()
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="balanced_accuracy")
clf = clf.fit(X_train,y_train)

# %%
y_pred = clf.predict(X_test)
# %%
BMAC = balanced_accuracy_score(y_test, y_pred)
print(f'BMAC: {BMAC}')
# %% Confusion matrix
conf_matrix = 0
if conf_matrix == 1:
   class_names = ['c_0','c_1','c_2']
   title = ["Confusion matrix"] 
   disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=class_names)
   disp.ax_.set_title(title)

   print(title)
   print(disp.confusion_matrix)
   plt.show()
# %% Try on real test data
X_test = pd.read_csv(f'{repopath}/task2/X_test.csv')
X_test = X_test.iloc[:,1:]

# print(X_test)
# X_test.describe()
X_test = pd.DataFrame(X_test, columns=X_red.columns)
X_test.describe()

# %%
y_pred = clf.predict(X_test)

# %%
submissions =  pd.DataFrame({
  'id': np.arange(0,len(y_pred)).astype(float),
  'y': y_pred
})
submissions.to_csv(f'{repopath}/prototyping/submission_svc.csv', index=False)

print(submissions)

# %%

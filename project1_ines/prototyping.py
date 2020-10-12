# Prototyping script

#%%
# import os
# os.getcwd()
# os.chdir('/Users/inespereira/Documents/Github/aml2020')

#%% Imports
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import logging

#%% Load training dataset from csv
X = pd.read_csv('X_train.csv')
X = X.iloc[:,1:]
y = pd.read_csv('y_train.csv')
y = y.iloc[:,1:]
logging.info('I have imported your training dataset! :D')

#%% Remove NaNs
def fill_nan(X, method):
  if method == 'mean':
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
  elif method == 'median':
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
  # elif method == 'freq': # TODO: consider whether we want this option as well.
  #   X = X.apply(lambda col: col.fillna(col.mode()), axis=0)
  return X

X = fill_nan(X,'mean')

X.describe()

#%% Outlier detection
# Good read: https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561
# Outliers can be univariate or multivariate. We need to assume they are multivariate
from sklearn.ensemble import IsolationForest
clf = IsolationForest(random_state=0).fit(X)
y_pred_train = clf.predict(X)

#%% Review results
a = np.array(np.where(y_pred_train==1)) # inliers
b = np.array(np.where(y_pred_train==-1)) # outliers
print(f"Here are a few examples of inliers:\n{a}")
print(f"The total number of inliers is: {a.size}")
print(f"Here are the detected outliers:\n{b}")
print(f"The total number of outliers is: {b.size}")
b = b.tolist()
print(type(b))
print(b)
outliers = [ item for elem in b for item in elem]
X.shape
X_inliers = X.drop(index=outliers)
# print(X_inliers)
# X_inliers.shape


#%% Feature selection
from main import feature_selection
# X_rfe, rfe = feature_selection(X,y,"rfe")
X_rfecv, rfecv = feature_selection(X,y,"rfecv")

#%%
# Just for rfecv
# print(rfecv.ranking_)
print(np.where(rfecv.ranking_==1))
# print(rfecv.grid_scores_)

#%% Decide later whether you want to include this.
# # Reference: https://www.scikit-yb.org/en/latest/api/model_selection/rfecv.html
# from yellowbrick.model_selection import RFECV as RFECV2
# model = Ridge()
# visualizer = RFECV2(model,step=20)
# visualizer.fit(X, y)        # Fit the data to the visualizer
# visualizer.show()           # Finalize and render the figure
# %%


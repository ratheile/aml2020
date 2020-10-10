# Dimensionality reduction script

#%%
# import os
# os.getcwd()
# os.chdir('/Users/inespereira/Documents/Github/aml2020')

#%% Imports
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import pandas as pd
import logging

#%% Load training dataset from csv
X = pd.read_csv('X_train.csv')
X = X.iloc[:,1:]
y = pd.read_csv('y_train.csv')
y = y.iloc[:,1:]
logging.info('I have imported your training dataset! :D')

#%%
estimator = Ridge()
selector = RFE(estimator, n_features_to_select=100, step=1, verbose=1)
selector = selector.fit(X, y)


# %%

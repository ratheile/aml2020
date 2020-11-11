#%% Imports
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

import os
repopath = '/Users/inespereira/Documents/Github/aml2020'
os.chdir(repopath)

#%% Load training dataset from csv
X = pd.read_csv(f'{repopath}/project3_ines/X_train.csv')
X = X.iloc[:,1:]
y = pd.read_csv(f'{repopath}/project3_ines/y_train.csv')
y = y.iloc[:,1:]
X_test = pd.read_csv(f'{repopath}/project3_ines/X_test.csv')
logging.info('I have imported your training dataset! :D')
print(f'Shape of training set is {X.shape}')

# %%
print(X)
X.describe()
print(y)
y.describe()

# %% Making class imbalance apparent
y['y'].value_counts()

# %% Get indices from the smaller classes
class1 = y.index[y['y'] == 1].tolist()
class2 = y.index[y['y'] == 2].tolist()
class3 = y.index[y['y'] == 3].tolist()

# %%
print(class1)
print(class3)

# %% Observations:
# - a lot of NaNs: but we probably need to extract features anyway
# - Class imbalance

# %% Plot some time series
n = 25
X.iloc[n,:].plot()
plt.show()
print(f'The corresponding class is: {y.iloc[n]}')
# %%

#%% Imports
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import peakutils # for basic peak-finding utils
import pywt # for wavelet transform bits
import wfdb # for physionet tools
import biosppy
import neurokit2 as nk

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
# print(X)
X.describe()
# print(y)
y.describe()

# %% Making class imbalance apparent
y['y'].value_counts()

# %% Get indices from the smaller classes
class1 = y.index[y['y'] == 1].tolist()
class2 = y.index[y['y'] == 2].tolist()
class3 = y.index[y['y'] == 3].tolist()

# %%
print(class2)

# %% Observations:
# - a lot of NaNs: but we probably need to extract features anyway
# - Class imbalance

# %% Plot some time series
n = 2
X.iloc[n,:].plot()
plt.show()
print(f'The corresponding class is: {y.iloc[n]}')
print(X.iloc[n,:])

#%%
ecg = X.iloc[n,:].dropna().to_numpy()
plt.plot(ecg)
plt.show()
type(ecg)
print(ecg)

#%% Biosppy
ecg_biosppy = biosppy.signals.ecg.ecg(
  signal=ecg, 
  sampling_rate=300, 
  show=True)

#%% Analyse biosppy summary
# type(ecg_biosppy)
# ecg_biosppy['ts']
# ecg_biosppy['rpeaks']
plt.plot(ecg_biosppy['filtered'])
plt.show()

# ecg_biosppy['templates_ts']
# ecg_biosppy['templates']
# ecg_biosppy['heart_rate_ts']
# ecg_biosppy['heart_rate']

# %% Wavelets
wavelets = pywt.wavedec(
  data=ecg,
  wavelet='db4', # from YouTube Video
  level=5
)

#%% Neurokit: has a bug and needs correction in source code or downgrading of sklearn
bio_features = nk.bio_process(
  ecg=ecg, 
  sampling_rate=300,
  ecg_filter_type='FIR',
  ecg_filter_band='bandpass',
  ecg_filter_frequency=[0.1,100],
  ecg_segmenter='hamilton',
)
# %% Plan: define the features you want to look at.
# 1. Preprocessing of data
#    1.1. (Ines) Removal of low frequency components. R peaks should be at the same height
#         + Smoothing of signal
#    1.3. (Raffi) Artefact removal (outlier removal): visuelle Überprüfung
#    1.4. (Francesco) Identify QRS complex and verify polarity
#         + Find features related to waves
# For an example: https://www.youtube.com/watch?v=WyjGCEWU4zY

# (Raffi) To isolate class 3: 
#    - Extract Fourier transform and plot mean for each class. You should see a peak
#    - Number of artefacts
# Detect inverted QRS and reinvert them?

# Data-driven: convolutional nets? A mean wave with the variance?

# References and software to check out:
# From AML TA: https://biosppy.readthedocs.io/en/stable/biosppy.signals.html#biosppy-signals-ecg
# Watch (general): https://www.youtube.com/watch?v=WyjGCEWU4zY
# https://pypi.org/project/py-ecg-detectors/
# Plotting: https://pypi.org/project/ecg-plot/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.electrocardiogram.html
# Time series anomalies: https://www.youtube.com/watch?v=qN3n0TM4Jno
# MATLAB code? https://www.youtube.com/watch?v=ofDHz77hm1c
# In MATLAB File exchange: https://ch.mathworks.com/matlabcentral/fileexchange/74537-ecg-analyser?s_tid=srchtitle
# Other in MATLAB FIle exchange: https://ch.mathworks.com/search.html?c%5B%5D=entire_site&q=ecg&page=1

# Other ideas: consider nested CV https://www.youtube.com/watch?v=DuDtXtKNpZs


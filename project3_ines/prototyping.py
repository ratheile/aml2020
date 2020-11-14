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
import scipy
import seaborn as sns
import heartpy

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
sampling_rate = 300

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
print(class3)

# %% Observations:
# - a lot of NaNs: but we probably need to extract features anyway
# - Class imbalance

#%% Allocate space for new training data
col_names = [
  'mean_HR',
  'std_HR',
  'P_waved'
]
new_X_train = pd.DataFrame(columns=col_names)
print(new_X_train)

# %% Plot some time series
# TODO: write for loop wrapper over this to apply preprocessing over all time series
n = 10
ecg = X.iloc[n,:].dropna().to_numpy()
plt.plot(ecg)
plt.show()
print(f'The corresponding class is: {y.iloc[n]}')
print(ecg)

# %% Apply Fourier transform: use this to exclude class 3 samples?
ecg_fft = np.fft.fft(ecg)
plt.plot(abs(ecg_fft))
plt.show()
type(ecg_fft)
print(ecg_fft)
#%% Biosppy
ecg_biosppy = biosppy.signals.ecg.ecg(
  signal=ecg, 
  sampling_rate=sampling_rate, 
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
ecg_biosppy['heart_rate']

# %% Populate new_X_train
new_X_train.loc[n,'mean_HR']=np.mean(ecg_biosppy['heart_rate'])
new_X_train.loc[n,'std_HR']=np.std(ecg_biosppy['heart_rate'])
print(new_X_train)

#%% Save filtered data to mat file
scipy.io.savemat('test.mat', {'mydata': ecg_biosppy['filtered']})

# %% Wavelets
wavelets = pywt.wavedec(
  data=ecg,
  wavelet='db4', # from YouTube Video
  level=5
)

# %% Neurokit2
#%% Also do analysis
df, info = nk.ecg_process(ecg_biosppy['filtered'], sampling_rate=sampling_rate)
analyze_df = nk.ecg_analyze(df, sampling_rate=sampling_rate)
analyze_df

#%%
df
#%% Download data
# ecg_signal = nk.data(dataset="ecg_3000hz")['ECG']
# ecg_signal = pd.Series(ecg_biosppy['filtered'],dtype='float64')
ecg_signal = pd.Series(df['ECG_Clean'],dtype='float64')

# Analyze the ecg signal
type(ecg_signal)
print(ecg_signal)
#%%

# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)

# Delineate
signal, waves = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate, method="dwt", show=True, show_type='all')


# %% Plan: define the features you want to look at.
# 1. Preprocessing of data
#    1.1. (Ines) Removal of low frequency components. R peaks should be at the same height
#         + Smoothing of signal → ecg function from biosppy does filtering.
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

# Pipeline steps
# 1. Detection and exclusion of class 3 from training set (TODO)
# 2. Detection of flipped signals and flipping (TODO)
# 3. Filtering (getting isoelectric line and smoothing)
# 4. Waveform detection
#   4.1 R-peaks and HR: mean_HR, std_HR
#   4.2 P, QRS and T: number of P waves, mean_QRS_amplitude, mean_QRS_duration, std_QRS_duration (TODO: analysis of analysis results to get these features)
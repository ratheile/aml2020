# %% Imports
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import peakutils  # for basic peak-finding utils
import pywt  # for wavelet transform bits
import wfdb  # for physionet tools
import biosppy
import neurokit2 as nk
import scipy
import seaborn as sns
import heartpy

import os
repopath = '/Users/inespereira/Documents/Github/aml2020'
os.chdir(repopath)

# %% Load training dataset from csv
X = pd.read_csv(f'{repopath}/project3_ines/X_train_small.csv')
# X = X.iloc[:, 1:]
y = pd.read_csv(f'{repopath}/project3_ines/y_train_small.csv')
# y = y.iloc[:, 1:]
X_u = pd.read_csv(f'{repopath}/project3_ines/X_test.csv')
logging.info('I have imported your training dataset! :D')
print(f'Shape of training set is {X.shape}')
sampling_rate = 300

# %%
X.iloc[:,17979]

# %% Save smaller dataset to not waste hours waiting for loading
NROWS = 30
X.iloc[:NROWS, :].to_csv('project3_ines/X_train_small.csv', index=False)
y.iloc[:NROWS, :].to_csv('project3_ines/y_train_small.csv', index=False)

# %%
NROWS = 10
X_u.iloc[:NROWS, :].to_csv('project3_ines/X_test_small.csv', index=False)

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

# %% Allocate space for new training data
col_names = [
    'mean_HR',
    'std_HR',
    'P_waved'
]
new_X_train = pd.DataFrame(columns=col_names)
print(new_X_train)

# %% Plot some time series
# TODO: write for loop wrapper over this to apply preprocessing over all time series
n = 3
ecg = X.iloc[n, :].dropna().to_numpy()
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
# %% Biosppy
ecg_biosppy = biosppy.signals.ecg.ecg(
    signal=ecg,
    sampling_rate=sampling_rate,
    show=True)

# %% Analyse biosppy summary
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
new_X_train.loc[n, 'mean_HR'] = np.mean(ecg_biosppy['heart_rate'])
new_X_train.loc[n, 'std_HR'] = np.std(ecg_biosppy['heart_rate'])
print(new_X_train)

# %% Save filtered data to mat file
scipy.io.savemat('test.mat', {'mydata': ecg_biosppy['filtered']})

# %% Wavelets
wavelets = pywt.wavedec(
    data=ecg,
    wavelet='db4',  # from YouTube Video
    level=5
)

# %% Neurokit2
# %% Also do analysis
df, info = nk.ecg_process(ecg_biosppy['filtered'], sampling_rate=sampling_rate)
analyze_df = nk.ecg_analyze(df, sampling_rate=sampling_rate)
analyze_df

# %%
df
# %% Download data
# ecg_signal = nk.data(dataset="ecg_3000hz")['ECG']
# ecg_signal = pd.Series(ecg_biosppy['filtered'],dtype='float64')
ecg_signal = pd.Series(df['ECG_Clean'], dtype='float64')

# Analyze the ecg signal
type(ecg_signal)
print(ecg_signal)
# %%

# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)

# Delineate
signal, waves = nk.ecg_delineate(
    ecg_signal, rpeaks, sampling_rate=sampling_rate, method="dwt", show=True, show_type='all')


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
# Watch (general): https://www.youtube.com/watch?v=WyjGCEWU4zY
# https://pypi.org/project/py-ecg-detectors/
# Plotting: https://pypi.org/project/ecg-plot/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.electrocardiogram.html
# Time series anomalies: https://www.youtube.com/watch?v=qN3n0TM4Jno
# Other ideas: consider nested CV https://www.youtube.com/watch?v=DuDtXtKNpZs

# Preprocessing pipeline steps
# 1. Detection and exclusion of class 3 from training set (TODO Raffi)
# 2. Detection of flipped signals and flipping (TODO Raffi + Inês)
# 3. Filtering (getting isoelectric line and smoothing)
# 4. Waveform detection
#   4.1 R-peaks and HR:
#         - mean_HR (class 1)
#         - std_HR (class 1)
#   4.2 P, QRS and T: TODO (Francesco)
#         - number of P waves, amplitude_P_wave (class 1)
#         - mean_S_amplitude, ?? std_S_amplitude (class 2)
#         - mean_QRS_duration, (class 2)
#         - std_QRS_duration (class 2)

# Build preprocessing pipeline TODO (Inês)
#    - Remove preprocessing from last project (PCA, outlier detection, standardization)
#    - Write down new Estimator
#
# Diagnostics plots: TODO (Raffi + Francesco): interactive, that show full time series and segmented P-QRS-T waves
# import plotly.express as px
#
# Sanity checks
#    - Check that HR and R-peaks between biosppy and Neurokit

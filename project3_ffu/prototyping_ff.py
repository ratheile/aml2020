#%% Imports
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

import os
repopath = '/Users/francescofusaro/Documents/Github/aml2020'
os.chdir(repopath)

from biosppy.signals import ecg
from ecgdetectors import Detectors
from hrv import HRV
#%% Load datasets
X = pd.read_csv(f'{repopath}/project3_ffu/X_train.csv')
X = X.iloc[:,1:]
y = pd.read_csv(f'{repopath}/project3_ffu/y_train.csv')
y = y.iloc[:,1:]
X_test = pd.read_csv(f'{repopath}/project3_ffu/X_test.csv')
X_test = X_test.iloc[:,1:]
logging.info('I have imported your training dataset! :D')
print(f'Shape of training set is {X.shape}')
# %% Create class specfic dataframes
class0_ls = y.index[y['y'] == 0].tolist() #helthy
class1_ls = y.index[y['y'] == 1].tolist() #Arrhythmia1
class2_ls = y.index[y['y'] == 2].tolist() #Arrhythmia2
class3_ls = y.index[y['y'] == 3].tolist() #Noise

Xseries_0 = X.iloc[class0_ls,:]
X_0 = pd.DataFrame(data=Xseries_0,columns=X.columns)

X_0.head(3)
#%% Define the sample rate (for all classes)
Fs = 300 # sampling rate in s^-1     
# %% Pick a random sample for class 0 for plotting
sample_id = np.random.randint(low=0, high=X_0.shape[0], size=1)
#sample_id = 1760 #class 0 example with high R-peak amplitude
#sample_id = 2730 #class 0 example with lower R-peak amplitude
#sample_id = 1591 #class 0 example with a noise
print(f'Picked sample {sample_id[0]} in class 0...')
sig_i = X_0.iloc[sample_id,:] #dataframe
#%% Transform signal to np array out of it
sig_i = sig_i.replace(to_replace='NaN',value=np.nan)
sig_i_np = (sig_i.to_numpy()).ravel()
print(type(sig_i_np))
# %% Remove nan from random sample
ls_0 = len(sig_i_np)
sig_i_np = sig_i_np[~np.isnan(sig_i_np)]
print(f'Removed {ls_0 - len(sig_i_np)} nan values...') 
# %% Plot the random sample
N = len(sig_i_np)  # number of samples
T = (N - 1) / Fs  # duration
ts = np.linspace(0, T, N, endpoint=False)  # relative timestamps
plt.plot(ts, sig_i_np, lw=1)
plt.xlabel('time (s)')
plt.ylabel('amplitude (mV?)')
plt.title(f'Sample {sample_id}')
plt.grid()
plt.show()
#%% Process the sample with biosspy ecg
out = ecg.ecg(signal=sig_i_np, sampling_rate=Fs, show=True)

# ts (array) – Signal time axis reference (seconds).
# filtered (array) – Filtered ECG signal.
# rpeaks (array) – R-peak location indices.
# templates_ts (array) – Templates time axis reference (seconds).
# templates (array) – Extracted heartbeat templates.
# heart_rate_ts (array) – Heart rate time axis reference (seconds).
# heart_rate (array) – Instantaneous heart rate (bpm).

(ts, filtered, rpeaks, templates_ts, 
  templates, heart_rate_ts, heart_rate) = out

# %% Extract R-peaks amplitude and some stats
rpeaks_amp = sig_i_np[rpeaks]
rpeaks_amp_sd = np.std(rpeaks_amp)
rpeaks_amp_median = np.median(rpeaks_amp)
rpeaks_amp_mean = np.mean(rpeaks_amp)
# %% rpeaks detection with py-ecg-detectors
# https://pypi.org/project/py-ecg-detectors/
detectors = Detectors(Fs)
#r_peaks_det = detectors.swt_detector(sig_i_np)
r_peaks_det = detectors.two_average_detector(sig_i_np)
plt.figure()
plt.plot(sig_i_np)
plt.plot(r_peaks_det, sig_i_np[r_peaks_det], 'ro')
plt.xlabel('sample id (-)')
plt.ylabel('amplitude (mV?)')
plt.title('Detected R-peaks')
plt.show()
# %% HR analysis with HRV
# https://pypi.org/project/py-ecg-detectors/
# https://github.com/berndporr/py-ecg-detectors/blob/master/hrv_time_domain_analysis.py
hrv_class = HRV(Fs) #init with sample frequency
hr = hrv_class.HR(r_peaks_det) #heart rate
nn20 = hrv_class.NN20(r_peaks_det) # the number of pairs of successive NNs (read RRs) that differ by more than 20 ms.
sdnn = hrv_class.SDNN(r_peaks_det) # calculate SDNN, the standard deviation of NN intervals.
# %% 

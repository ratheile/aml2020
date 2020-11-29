# %%
# should provide auto reload
# %load_ext autoreload
# %autoreload 2
# Path hack.import sys, os

# %%
sys.path.insert(0, os.path.abspath('..'))

from pywt import wavedec, waverec
import scipy.signal as sig

import heartpy as hp
import neurokit2 as nk
from biosppy.signals import ecg as bspy 

import numpy as np
import pandas as pd
from copy import deepcopy
from modules import ConfigLoader
from project2_raffi.visualization import pca

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def nk2_ecg_process_AML(ecg_signal, sampling_rate=300):
  """Process an ECG signal as original neurokit2 function, see:
  https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/ecg/ecg_process.html#ecg_process
  
  However, to increase flexibility, 'method' parameter is specifically set for each subfunction call:
  
  - ecg_clean methods: Can be one of ‘neurokit’ (default), ‘biosppy’, ‘pamtompkins1985’, ‘hamilton2002’, ‘elgendi2010’, ‘engzeemod2012’.
  
  - ecg_peaks methods: Can be one of ‘neurokit’ (default), ‘pamtompkins1985’, ‘hamilton2002’, ‘christov2004’, ‘gamboa2008’, ‘elgendi2010’, ‘engzeemod2012’ or ‘kalidas2017’
  
  - ecg_delineate methods: Indentify PQRST peak Can be one of ‘peak’ (default) for a peak-based method, ‘cwt’ for continuous wavelet transform or ‘dwt’ for discrete wavelet transform.
  see: https://neurokit2.readthedocs.io/en/latest/examples/ecg_delineate.html
  """
  
  # Filtering and smoothing
  ecg_preprocess_clean_method = 'biosppy' # TODO: from base_cfg
  ecg_cleaned = nk.ecg.ecg_clean(
    ecg_signal,
    sampling_rate=sampling_rate,
    method=ecg_preprocess_clean_method
    )
  
  # R-peak detection
  # Following example from: https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg.ecg_rsp
  ecg_preprocess_rpeaks_method = 'neurokit' # TODO: from base_cfg
  rpeaks, rpeaks_info = nk.ecg.ecg_peaks(
    ecg_cleaned=ecg_cleaned,
    sampling_rate=sampling_rate,
    method=ecg_preprocess_rpeaks_method,
    correct_artifacts=True
  )

  rate = nk.signal_rate(
    rpeaks,
    sampling_rate=sampling_rate,
    desired_length=len(ecg_cleaned)
    )

  # Quality evaluation: https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg.ecg_quality
  quality = nk.ecg.ecg_quality(ecg_cleaned, rpeaks=None, sampling_rate=sampling_rate)

  # Create new dataframe with raw and clean ECG, as well as rate and quality
  signals = pd.DataFrame({"ECG_Raw": ecg_signal, "ECG_Clean": ecg_cleaned, "ECG_Rate": rate, "ECG_Quality": quality})

  # Additional info of the ecg signal
  ecg_preprocess_delineate_method = 'dwt'
  delineate_signal, delineate_info = nk.ecg.ecg_delineate(
    ecg_cleaned=ecg_cleaned,
    rpeaks=rpeaks_info,
    sampling_rate=sampling_rate,
    method=ecg_preprocess_delineate_method
  )
  
  if ecg_preprocess_delineate_method != 'peak':
    # 'dwt' and 'cwt' Unlike the peak method, 'dwt' and 'cwt' does not idenfity the Q-peaks and S-peaks.
    delineate_signal_peak, delineate_info = nk.ecg.ecg_delineate(
    ecg_cleaned=ecg_cleaned, 
    rpeaks=rpeaks_info, 
    sampling_rate=sampling_rate, 
    method='peak'
    )
    delineate_signal['ECG_Q_Peaks'] = delineate_signal_peak['ECG_Q_Peaks']
    delineate_signal['ECG_S_Peaks'] = delineate_signal_peak['ECG_S_Peaks']
      
  cardiac_phase = nk.ecg.ecg_phase(
    ecg_cleaned=ecg_cleaned,
    rpeaks=rpeaks_info,
    delineate_info=delineate_info
    )
  # Reconstructing ouput identical to nk.ecg_process We need this to use nk.ecg_analyze
  signals = pd.concat([signals, rpeaks, delineate_signal, cardiac_phase], axis=1)

  return signals, rpeaks_info

# %% some local testing:
env_cfg = ConfigLoader().from_file('../env/env.yml')
# %%
print(env_cfg)

# %%
df_X = pd.read_csv(f"{env_cfg['datasets/project3/path']}/X_train.csv")
df_y = pd.read_csv(f"{env_cfg['datasets/project3/path']}/y_train.csv")
df_X_u = pd.read_csv(f"{env_cfg['datasets/project3/path']}/X_test.csv")  # unlabeled

#%%
sample_rate = 300
def time_scale(samples): return np.arange(samples.shape[0]) / sample_rate
def vec_to_ind(vec_0_1): return vec_0_1[vec_0_1 == 1].index.tolist()
# %%
X = df_X.iloc[:, 1:]
y = df_y.iloc[:, 1:].values.ravel()
X_u = df_X.iloc[:, 1:]

X_124 = X.iloc[y != 3]
X_3 = X.iloc[y == 3]
# crv = X_124.iloc[13, ]
# crv = X_124.iloc[10, ] # T peaks onto R peaks
# crv = X_124.iloc[11, ] # Filtering demo 
# crv = X_124.iloc[3, ]
crv = X_3.iloc[2,]
crv = crv[~np.isnan(crv.values.astype(float))]

crv_df = pd.DataFrame({
    'v': crv.values,
    't': time_scale(crv)
})

px.line(crv_df, x='t', y='v')
crv = crv.values.astype(float)



#%% Biosppy filtering
out = bspy.ecg(signal=crv, sampling_rate=sample_rate, show=False)
(ts, filtered_biosppy, rpeaks_biosppy, templates_ts,
templates, heart_rate_ts, heart_rate) = out


#%% neurokit filtering
signals, info = nk2_ecg_process_AML(crv)
# Find peaks
ecg_orig_clean = nk.ecg_clean(crv)
peaks, info = nk.ecg_peaks(ecg_orig_clean, sampling_rate=sample_rate)
# Compute HRV indices
# fig1 = nk.hrv(peaks, sampling_rate=sample_rate, show=True)
fig1 = nk.ecg_plot(signals, sampling_rate=sample_rate, show_type='default') 
fig1.set_size_inches(18.5, 10.5)
# fig2 = nk.ecg_plot(signals, sampling_rate=sample_rate, show_type='artifacts') 
# f, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 12))
# ax.plot(list(range(len(ecg_orig_clean))), ecg_orig_clean)



# heartbeats_pivoted = heartbeats_2.pivot(index="Time", columns="Label", values="Signal")


filtered_nk2 = signals['ECG_Clean'].to_numpy().ravel()
rate_nk2 = signals['ECG_Rate']
quality_nk2 = signals['ECG_Quality']

########################### Additional Filtering ##############################
# %% heartpy find anomal peaks
hp_sig = ecg_orig_clean
hp_sig = hp.remove_baseline_wander(hp_sig, sample_rate)
hp_sig = hp.scale_data(hp_sig)
# hp_sig = hp.filter_signal(hp_sig, 0.05, sample_rate, filtertype='highpass')
wd, m = hp.process(hp_sig, sample_rate)
# visualise in plot of custom size
plt.figure(figsize=(12, 4))
hp.plotter(wd, m)
wd_rejected_x = wd['removed_beats']

# %% filtering using the above
# find peaks who deviate from the median peak
filter_mask = np.ones(crv.shape[0], dtype=np.bool_)
rpeaks = vec_to_ind(signals['ECG_R_Peaks'])
rpeak_vals = crv[rpeaks] 
rpeak_tshd = np.median(rpeak_vals) + rpeak_vals.std()

# select peaks who are way above the median
outlier_rejected_x, _ = sig.find_peaks(crv, height=rpeak_tshd)

# select first 2 and last 2 peaks
n_border_peaks = 3
first_rpeaks = rpeaks[0:n_border_peaks]
last_rpeaks = rpeaks[-n_border_peaks:]

# concatinate different faulty peak sources
# may include the same peak twice
all_rejected = np.concatenate(
  (wd_rejected_x, outlier_rejected_x,
  first_rpeaks, last_rpeaks)
)
all_rejected = np.sort(all_rejected)

delta = 50
regions = np.zeros((len(all_rejected),2), dtype=np.int)
regions[:,0] = all_rejected - delta
regions[:,1] = all_rejected + delta

p_ons = vec_to_ind(signals['ECG_P_Onsets'])
t_off = vec_to_ind(signals['ECG_T_Offsets'])

# def rolling_window(a, window):
#   shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
#   strides = a.strides + (a.strides[-1],)
#   return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def set_to_closest(indicies, in_arr, to_the):
  out = np.zeros((len(indicies)))
  for i, ind in enumerate(indicies):
    dist = in_arr - ind
    if to_the == 'left':
      k = np.where(dist < 0, dist, -np.inf).argmax() # largest negative
    elif to_the == 'right': # right
      k = np.where(dist > 0, dist, np.inf).argmin() # smallest positive
    # return element if nothing found, otherwise set to bound
    out[i] = ind if np.abs(k) == np.inf else in_arr[k]
  return out


regions[:,0] = set_to_closest(regions[:,0], p_ons, to_the='left')
regions[:,1] = set_to_closest(regions[:,1], t_off, to_the='right')
# shrink mask by eps to mask the boundary points
eps = 5
regions[:,0] = regions[:,0] - eps
regions[:,1] = regions[:,1] + eps

regions = np.unique(regions, axis=0)
for i in range(regions.shape[0]):
  filter_mask[np.arange(*regions[i,:])] = False



# Prolong filtered region if we see successive 
# faulty peaks in a short time window
# filter_mask_rw = rolling_window(filter_mask, 2)
fmt = np.where(np.logical_xor(np.roll(filter_mask, 1), filter_mask))[0]
fmt = fmt[filter_mask[fmt] == False]

eps = 400
regions = np.zeros((len(fmt) - 1,2), dtype=np.int)
regions[:,0] = np.roll(fmt,1)[1:]
regions[:,1] = fmt[1:]
regions = regions[regions[:,1] - regions[:,0] < eps]
for i in range(regions.shape[0]):
  filter_mask[np.arange(*regions[i,:])] = False

change = sig.savgol_filter(hp_sig, window_length=11, polyorder=2, deriv=1) 


########################### Segment processor ##############################
#%%
signal_names = ['ECG_R_Peaks',
 'ECG_T_Peaks', 'ECG_T_Onsets', 'ECG_T_Offsets', 'ECG_P_Peaks',
 'ECG_P_Onsets', 'ECG_P_Offsets', 'ECG_R_Onsets', 'ECG_R_Offsets',
 'ECG_Q_Peaks', 'ECG_S_Peaks']

hb_peaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
heartbeats = nk.ecg_segment(signals["ECG_Clean"], hb_peaks, sample_rate)
heartbeats_2 = nk.epochs.epochs_to_df(heartbeats)

show_n_epochs = 5
epoch_mask = np.zeros((crv.shape[0],show_n_epochs), dtype=np.bool_)

for i in range(show_n_epochs):
  epoch_mask[heartbeats[str(i+1)]['Index'].values, i] = True

#%% hearbeat mask for francesco
rpeaks_0_1 = signals['ECG_R_Peaks']
rpeaks_window_dict = {}
for key, df in heartbeats.items():
  epoch_mask = np.zeros((crv.shape[0]), dtype=np.bool_)
  epoch_mask[df['Index'].values] = True
  peak_ind = np.logical_and(epoch_mask, filter_mask)
  peak_ind = np.logical_and(peak_ind, rpeaks_0_1)
  where = np.where(peak_ind)[0]
  if len(where) > 0:
    assert len(where) == 1
    rpeaks_window_dict[where[0]] = (key, df)

#%% ###################### Extract Segment Lengths for one signal #################
#apply filter mask to our signal
sig_filtered= signals[filter_mask]
killed_idx_ratio = 1-len(sig_filtered)/len(signals)
#TODO: kill the sample if good to bad is below 10%?

#extract indexes
r_f = np.array(vec_to_ind(sig_filtered['ECG_R_Peaks']))
p_ons_f = np.array(vec_to_ind(sig_filtered['ECG_P_Onsets']))
p_f = np.array(vec_to_ind(sig_filtered['ECG_P_Peaks']))
p_off_f = np.array(vec_to_ind(sig_filtered['ECG_P_Offsets']))
r_ons_f = np.array(vec_to_ind(sig_filtered['ECG_R_Onsets']))
q_f = np.array(vec_to_ind(sig_filtered['ECG_Q_Peaks']))
s_f = np.array(vec_to_ind(sig_filtered['ECG_S_Peaks']))
r_off_f = np.array(vec_to_ind(sig_filtered['ECG_R_Offsets']))
t_ons_f = np.array(vec_to_ind(sig_filtered['ECG_T_Onsets']))
t_f = np.array(vec_to_ind(sig_filtered['ECG_T_Peaks']))
t_off_f = np.array(vec_to_ind(sig_filtered['ECG_T_Offsets']))

#%%
#loop through each epoch (r_peaks) and extract corresponding secondary peaks
###################### Extract Segment Lengths for one signal #################
###Preparations and apply filter mask
#define array where to store the data
#|0|1|2|3|4|5|6|7|8|9|10|
#|p_ons|p|p_off|r_ons|q|r|s|r_off|t_ons|t|t_off|
def init_arrays():
  n_peak_types = 11
  ep_idx_init = np.array([np.nan]*n_peak_types)
  ep_i_dist_init = np.array([np.nan]*55)
  
  return ep_idx_init, ep_i_dist_init

def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return array[idx]

def pick_a_peak(summary,idx,r_peak,sec_peaks_vec,to_the,window_bounds):
  
  #check for candidates
  if to_the == 'left':
    v = sec_peaks_vec[np.logical_and(np.greater_equal(r_peak,sec_peaks_vec),np.less(window_bounds[0],sec_peaks_vec))]
  elif to_the == 'right':
    v = sec_peaks_vec[np.logical_and(np.less_equal(r_peak,sec_peaks_vec),np.greater(window_bounds[1],sec_peaks_vec))]

  if len(v) > 0:
      #find the the nearest
      p_idx = find_nearest(sec_peaks_vec,r_peak)
      summary[idx]=p_idx
    
  return summary

def calc_distances(ep_i_p_idx):
  #p_ons
  #|p|p_off|r_ons|q|r|s|r_off|t_ons|t|t_offs|  0:9 len10
  #p
  #|p_off|r_ons|q|r|s|r_off|t_ons|t|t_offs|  0:8 len9
  #r_ons
  #|q|r|s|r_off|t_ons|t|t_offs|  0:7 len8
  #q
  #|r|s|r_off|t_ons|t|t_offs|  0:6 len7
  #r
  #|s|r_off|t_ons|t|t_offs|  0:5 len6
  #s
  #|r_off|t_ons|t|t_offs|  0:3 len4
  #r_off
  #|t_ons|t|t_offs|  0:2 len3
  #t_ons
  #|t|t_offs|  0:1 len2
  #t
  #|t_offs|  0 len1
  #10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 55
  distances = np.array([])
  for i in range(len(ep_i_p_idx)-1):
    dist_i = np.subtract(ep_i_p_idx[i],ep_i_p_idx[i+1:])
    distances = np.append(distances,dist_i)
  
  return distances

def rpeaks_window_bounds(r_peak_id,rpeaks_window_dict,eps):
    _, df_wind = rpeaks_window_dict[r_peak_id]
    lb = df_wind.Index.min() - eps
    ub = df_wind.Index.max() + eps
    
    return np.array([lb,ub])
    
E= np.zeros(shape=(len(r_f),55)) # r_peaks * 55 distances measure in each epoch
for i in range(len(r_f)):
  init_ep_i_idx, _ = init_arrays()
  ep_i_p_idx = init_ep_i_idx
  window_bounds = rpeaks_window_bounds(r_f[i],rpeaks_window_dict,50)
  #to the left
  sec_peaks_left = [p_ons_f,p_f,p_off_f,r_ons_f,q_f]
  for idx, sec_peak_v in enumerate(sec_peaks_left):
      ep_i_p_idx = pick_a_peak(ep_i_p_idx,idx,r_f[i],sec_peak_v,'left', window_bounds)
  
  ep_i_p_idx[5]=r_f[i]
  
  #to the right
  sec_peaks_right = [s_f,r_off_f,t_ons_f,t_f,t_off_f]
  idx_offset = 6
  for idx, sec_peak_v in enumerate(sec_peaks_right):
      ep_i_p_idx = pick_a_peak(ep_i_p_idx,idx+idx_offset,r_f[i],sec_peak_v,'right',window_bounds)
        
  #TODO: count the nans and reject the epoch if too many are missing?
  #compute the distances
  #if np.np.sum(np.isnan(ep_i_p_idx[0:5])) >= 2 and np.sum(np.isnan(ep_i_p_idx[6:10])) >= 2:
  #compute the distance
  #else:
  #set to nan
    
  #compute the distances in the epoch
  if np.all(np.diff(ep_i_p_idx[~np.isnan(ep_i_p_idx)])) > 0:
    ep_i_dist = calc_distances(ep_i_p_idx)
  else:
    #logging.warning(f'secondary peak detection for epoch {i} failed')
    _, ep_i_dist_init = init_arrays()
    ep_i_dist = ep_i_dist_init
    
  #store and go to the next epoch
  E[i]=-ep_i_dist[:]

# aggregate into dataframe and compute mean and std for the signal
E_df = pd.DataFrame(data=E) 
#%%
########################### Plotting ##############################
fig = make_subplots(rows=3, cols=1,
  shared_xaxes='rows',
  row_heights=[0.5, 0.25, 0.25])
time_ax = time_scale(crv)

fig.update_layout(
  margin={ 'l': 40, 'b': 40, 't': 10, 'r': 0 },
  hovermode='closest',
  height=1200,
  width=1500,
  yaxis2=dict(
    rangemode='tozero'
  )
)

fig.add_trace(
    go.Scatter(x=time_ax, y=crv, name='raw_signal'),
    row=1, col=1
)
for i in range(show_n_epochs):
  fig.add_trace(
      go.Scatter(x=time_ax, y=epoch_mask[:,i].astype(float) * 1000, name='epoch_mask'),
      row=1, col=1
  )
fig.add_trace(
    go.Scatter(x=time_ax, y=filtered_biosppy, name='filtered_bspy'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=time_ax, y=filtered_nk2, name='filtered_nk2'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=time_ax, y=rate_nk2, name='rate_nk2'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=time_ax, y=change, name='rate_nk2'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=time_ax, y=filter_mask.astype(float), name='filter_mask'),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=time_ax, y=quality_nk2, name='quality_nk2'),
    row=3, col=1
)


def plot_points(signals=None, metric_name=None,
  index_array=None,
  y=crv, color='black', size=6,
  mask=None, color_false='red',
  row=1):
  
  if index_array is None:
    metric = signals[metric_name]
    marker_index = metric[metric == 1].index.tolist()
  else:
    marker_index = index_array

  marker_ts = time_ax[marker_index]


  args = dict(
    y=y[marker_index],
    name=metric_name, mode='markers',
    marker=dict(
    color=color,
    size=size,
    line=dict(
      color='MediumPurple',
      width=2
    )
    )
  )

  if mask is not None:
    marker_mask = mask[marker_index]
    colors = [color if i == True else color_false for i in marker_mask]
    args['colors'] = colors

  fig.add_trace(
    go.Scatter(x=marker_ts, **args),
    row=row, col=1
  )

plot_points(signals, 'ECG_R_Peaks', color='black')
plot_points(signals, 'ECG_R_Onsets', color='green') # Q_peak onset
plot_points(signals, 'ECG_R_Offsets', color='green') # S_peak onset
plot_points(signals, 'ECG_Q_Peaks', color='green')

plot_points(signals, 'ECG_R_Peaks',y=quality_nk2, color='green', row=3)
plot_points(signals, 'ECG_R_Peaks',y=rate_nk2, color='green', row=2)

plot_points(signals, 'ECG_S_Peaks', color='blue')

plot_points(signals, 'ECG_P_Peaks', color='orange')
plot_points(signals, 'ECG_P_Onsets', color='orange')

plot_points(signals, 'ECG_T_Peaks', color='red')
plot_points(signals, 'ECG_T_Offsets', color='red')

p_ons = vec_to_ind(signals['ECG_P_Onsets'])
t_off = vec_to_ind(signals['ECG_T_Offsets'])
plot_points(signals, 'ECG_P_Onsets', color='orange', y=quality_nk2, row=3 )
plot_points(signals, 'ECG_T_Offsets', color='red', y=quality_nk2, row=3)

plot_points(index_array=all_rejected, metric_name='Outlier_Peaks', size=15)
plot_points(index_array=fmt, metric_name='Outlier_Peaks', y=filter_mask.astype(float), size=5, row=3)
fig
# %%

# %%

# %%
# %%

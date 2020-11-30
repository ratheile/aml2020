import numpy as np
import neurokit2 as nk
import pandas as pd

def rpeaks_window(signals, filter_mask, sample_rate):
  #%% hearbeat mask for francesco
  signal_clean = signals["ECG_Clean"]
  hb_peaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
  heartbeats = nk.ecg_segment(signal_clean, hb_peaks, sample_rate)
  rpeaks_0_1 = signals['ECG_R_Peaks']
  rpeaks_window_dict = {}
  for key, df in heartbeats.items():
    epoch_mask = np.zeros((signal_clean.shape[0]), dtype=np.bool_)
    unbounded_mask = df['Index'].values
    bounded_mask = unbounded_mask[unbounded_mask < signal_clean.shape[0]]
    bounded_mask= bounded_mask[bounded_mask >= 0]
    epoch_mask[bounded_mask] = True
    peak_ind = np.logical_and(epoch_mask, filter_mask)
    peak_ind = np.logical_and(peak_ind, rpeaks_0_1)
    where = np.where(peak_ind)[0]
    if len(where) > 0:
      for p in where: 
        rpeaks_window_dict[p] = (key, df)
  return rpeaks_window_dict


def vec_to_ind(vec_0_1): return vec_0_1[vec_0_1 == 1].index.tolist()

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


def clean_signal_bounds(signals):
  lb = 0
  ub = max(signals.index)
  return np.array([lb,ub])


#%% create dataframe column names
def dist_df_colnames(app):
  #|0|1|2|3|4|5|6|7|8|9|10|
  #|p_ons|p|p_off|r_ons|q|r|s|r_off|t_ons|t|t_off|
  nk_names = ['ECG_P_Onsets','ECG_P_Peaks','ECG_P_Offsets',
              'ECG_R_Onsets','ECG_Q_Peaks','ECG_R_Peaks',
              'ECG_S_Peaks','ECG_R_Offsets','ECG_T_Onsets',
              'ECG_T_Peaks', 'ECG_T_Offsets']
  new_names = []
  for n in range(0,len(nk_names)-1):
    for i in range(n+1,len(nk_names)):
      new_names.append("".join(nk_names[n]+'.'+nk_names[i]+app))
  return new_names


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


def pick_a_peak(summary,idx,r_peak,sec_peaks_vec,to_the,window_bounds,sb):
  #TODO: what needs to be done with signal bound?
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



def calc_interconnection_summary(signals, sampling_rate, filter_mask):
  rpeaks_window_dict = rpeaks_window(signals, filter_mask, sampling_rate)

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

  E= np.zeros(shape=(len(r_f),55)) # r_peaks * 55 distances measure in each epoch

  for i in range(len(r_f)):
    init_ep_i_idx, _ = init_arrays()
    ep_i_p_idx = init_ep_i_idx
    window_bounds = rpeaks_window_bounds(r_f[i],rpeaks_window_dict,50)
    signal_bounds = clean_signal_bounds(signals)

    #to the left
    sec_peaks_left = [p_ons_f,p_f,p_off_f,r_ons_f,q_f]

    for idx, sec_peak_v in enumerate(sec_peaks_left):
        ep_i_p_idx = pick_a_peak(ep_i_p_idx,idx,r_f[i],sec_peak_v,'left', window_bounds,signal_bounds)
    
    ep_i_p_idx[5]=r_f[i]
    
    #to the right
    sec_peaks_right = [s_f,r_off_f,t_ons_f,t_f,t_off_f]
    idx_offset = 6

    for idx, sec_peak_v in enumerate(sec_peaks_right):
        ep_i_p_idx = pick_a_peak(ep_i_p_idx,idx+idx_offset,r_f[i],sec_peak_v,'right',window_bounds, signal_bounds)
          
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

  #%% consolidate the distance data frame
  mean_ls = []
  std_ls = []

  for col_id in E_df.columns:
    mean = E_df.iloc[:,col_id].mean()
    std = E_df.iloc[:,col_id].std()
    mean_ls.append(mean)
    std_ls.append(std)

  mean_ls.extend(std_ls)
  res = np.array(mean_ls)
  col_name = dist_df_colnames('.mean')
  col_name_std = dist_df_colnames('.std')
  col_name.extend(col_name_std)
  # Dist_i_df = pd.DataFrame([res],columns=col_name)

  return res, col_name

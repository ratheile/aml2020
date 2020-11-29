import logging

import numpy as np
import pandas as pd
import warnings

import heartpy as hp
import neurokit2 as nk
from biosppy.signals import ecg as bspy 

import scipy.signal as sig

def vec_to_ind(vec_0_1): return vec_0_1[vec_0_1 == 1].index.tolist()

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



def heartpy_filter(hp_sig, sample_rate):
  with warnings.catch_warnings():
    # The maximal number of iterations maxit (set to 20 by the program)
    # allowed for finding a smoothing spline with fp=s has been reached: s
    # too small.
    # There is an approximation returned but the corresponding weighted sum
    # of squared residuals does not satisfy the condition abs(fp-s)/s < tol.
    warnings.simplefilter("ignore", category=UserWarning)
    # RuntimeWarning: Mean of empty slice.
    warnings.simplefilter("ignore", category=RuntimeWarning)
    hp_sig = hp.remove_baseline_wander(hp_sig, sample_rate)
    hp_sig = hp.scale_data(hp_sig)
    wd, m = hp.process(hp_sig, sample_rate)
    wd_rejected_x = wd['removed_beats']
    return wd_rejected_x

# Requirements
# signals come from signals, info = nk2_ecg_process_AML(crv)
def create_filter_mask(crv, signals, sample_rate):
  wd_rejected_x = None
  try:
    with warnings.catch_warnings():
      # RuntimeWarning: Mean of empty slice.
      warnings.simplefilter("ignore", category=RuntimeWarning)
      ecg_orig_clean = nk.ecg_clean(crv)
    wd_rejected_x = heartpy_filter(ecg_orig_clean, sample_rate)
  except hp.exceptions.BadSignalWarning:
    logging.error('Heartpy rejection filtering crashed')
    
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
  

  if wd_rejected_x is not None:
    all_rejected = np.concatenate(
      (wd_rejected_x, outlier_rejected_x,
      first_rpeaks, last_rpeaks)
    )
  else:
    all_rejected = np.concatenate(
      (outlier_rejected_x,
      first_rpeaks, last_rpeaks)
    )



  all_rejected = np.sort(all_rejected)

  delta = 50
  regions = np.zeros((len(all_rejected),2), dtype=np.int)
  regions[:,0] = all_rejected - delta
  regions[:,1] = all_rejected + delta

  p_ons = vec_to_ind(signals['ECG_P_Onsets'])
  t_off = vec_to_ind(signals['ECG_T_Offsets'])

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

  eps = 500
  regions = np.zeros((len(fmt) - 1,2), dtype=np.int)
  regions[:,0] = np.roll(fmt,1)[1:]
  regions[:,1] = fmt[1:]
  regions = regions[regions[:,1] - regions[:,0] < eps]
  for i in range(regions.shape[0]):
    filter_mask[np.arange(*regions[i,:])] = False

  return filter_mask



def quality_check_old(sig, sample_rate):
  # exclude valid low quality segments

  filter_mask = np.ones(sig.shape[0])
  try:
    success = False
    ecg_orig_clean = nk.ecg_clean(sig)
    filtered = hp.filter_signal(ecg_orig_clean, 0.05, sample_rate, filtertype='notch')
    hp_sig = hp.remove_baseline_wander(filtered, sample_rate)
    hp_sig = hp.scale_data(hp_sig)
    wd, m = hp.process(hp_sig, sample_rate)
    logging.warning('quality check succeded')
    success = True
  except hp.exceptions.BadSignalWarning:
    success = False

  return filter_mask, success
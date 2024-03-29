# Imports
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from biosppy.signals import ecg
#from ecgdetectors import Detectors
#from hrv import HRV
import neurokit2 as nk
from scipy.signal import resample

import heartpy as hp
from joblib import Parallel, delayed  
from tqdm import tqdm  
from preproc_viewer import create_app
from .filter import create_filter_mask
from .extended_features import calc_interconnection_summary, dist_df_colnames

# from pandas_profiling import ProfileReport
# from pathlib import Path
# Split Classes
def split_classes(X,y):
  class0_ls = y.index[y['y'] == 0].tolist() #healthy
  class1_ls = y.index[y['y'] == 1].tolist() #Arrhythmia1
  class2_ls = y.index[y['y'] == 2].tolist() #Arrhythmia2
  class3_ls = y.index[y['y'] == 3].tolist() #Noise
  
  X0 = X.iloc[class0_ls,:]
  X1 = X.iloc[class1_ls,:]
  X2 = X.iloc[class2_ls,:]
  X3 = X.iloc[class3_ls,:]
  
  return(X0, X1, X2, X3)

# Define more flexible ecg_process function
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
      
# Extracted peaks summary
def calc_peak_summary(signals, sampling_rate, filter_mask):
#  method which calculates a subset on entries in feature list  
#  feature_list = ['ECG_Quality_Mean', 'ECG_Quality_STD',
#                    'ECG_Rate_Mean', 'ECG_HRV',
#                    'R_P_biosppy', 
#                    'P_P/R_P', 'Q_P/R_P', 'R_P_neurokit' , 'S_P/R_P', 'T_P/R_P', >> in this function (5)
#                    'P_Amp_Mean', 'P_Amp_STD',    >> in this function (2)
#                    'Q_Amp_Mean', 'Q_Amp_STD',    >> in this function (2)
#                    'R_Amp_Mean', 'R_Amp_STD',    >> in this function (2)
#                    'S_Amp_Mean', 'S_Amp_STD',    >> in this function (2)
#                    'T_Amp_Mean', 'T_Amp_STD',    >> in this function (2)
#                    'QRS_t_Mean', 'QRS_t_STD',    >> in this function (2)
#                    'PR_int_Mean', 'PR_int_STD'   >> in this function (2)
#                    'PR_seg_Mean', 'PR_seg_STD',  >> in this function (2)                
#                    'QT_int_Mean', 'QT_int_STD']  >> in this function (2)
#                    'ST_seg_Mean', 'ST_seg_STD']  >> in this function (2)
   
      
  feature_names = [
        'P_P/R_P', 
        'Q_P/R_P', 
        'R_P_neurokit' , 
        'S_P/R_P', 
        'T_P/R_P',  #relative number of peaks TODO
        'P_Amp_Mean', 'P_Amp_STD', 
        'Q_Amp_Mean', 'Q_Amp_STD',
        'R_Amp_Mean', 'R_Amp_STD',
        'S_Amp_Mean', 'S_Amp_STD',
        'T_Amp_Mean', 'T_Amp_STD',
      ]
  
  # Peak summary
  summary = [] # needs to have len=25((2*10 = 20) + 5) at the end
  sig_pp = signals[signals['ECG_P_Peaks'] == 1]
  p_count= len(sig_pp)
  sig_qq = signals[signals['ECG_Q_Peaks'] == 1]
  q_count = len(sig_qq)
  sig_rr = signals[signals['ECG_R_Peaks'] == 1]
  r_count = len(sig_rr)
  sig_ss = signals[signals['ECG_S_Peaks'] == 1]
  s_count = len(sig_ss)
  sig_tt = signals[signals['ECG_T_Peaks'] == 1]
  t_count = len(sig_tt)
  
  # Peak counts
  p_rel = p_count/r_count
  q_rel = q_count/r_count
  s_rel = s_count/r_count
  t_rel = t_count/r_count
  summary.append(p_rel)
  summary.append(q_rel)
  summary.append(r_count)
  summary.append(s_rel)
  summary.append(t_rel)
  
  # Peak p amplitude
  if len(sig_pp) > 0:
    p_mean = sig_pp['ECG_Clean'].mean()
    p_std = sig_pp['ECG_Clean'].std()
  else:
    p_mean = np.nan
    p_std = np.nan
  summary.append(p_mean)
  summary.append(p_std)
  
  # Peak q amplitude
  if len(sig_qq) > 0:
    q_mean = sig_qq['ECG_Clean'].mean()
    q_std = sig_qq['ECG_Clean'].std()
  else:
    q_mean = np.nan
    q_std = np.nan
  summary.append(q_mean)
  summary.append(q_std)
  
  # Peak r amplitude
  if len(sig_rr) > 0:
    r_mean = sig_rr['ECG_Clean'].mean()
    r_std = sig_rr['ECG_Clean'].std()
  else:
    r_mean = np.nan
    r_std = np.nan
  summary.append(r_mean)
  summary.append(r_std)
  
  # Peak s amplitude
  if len(sig_ss) > 0:
    s_mean = sig_ss['ECG_Clean'].mean()
    s_std = sig_ss['ECG_Clean'].std()
  else:
    s_mean = np.nan
    s_std = np.nan
  summary.append(s_mean)
  summary.append(s_std)
  
  # Peak t amplitude
  if len(sig_tt) > 0:
    t_mean = sig_tt['ECG_Clean'].mean()
    t_std = sig_tt['ECG_Clean'].std()
  else:
    s_mean = np.nan
    s_std = np.nan
  summary.append(t_mean)
  summary.append(t_std)
  
  # check whether the signal is flipped
  is_flipped = False
  if (
    (len(sig_qq) > 0 and len(sig_rr) > 0 and
    abs(sig_qq['ECG_Clean'].mean()) > abs(sig_rr['ECG_Clean'].mean())) or
    (len(sig_ss) > 0 and len(sig_rr) > 0 and
    abs(sig_ss['ECG_Clean'].mean()) > abs(sig_rr['ECG_Clean'].mean()))
  ):
    is_flipped = True


  icon_values, icon_col_names = calc_interconnection_summary(
    signals, sampling_rate, filter_mask
  )
  
  summary.extend(icon_values)
  feature_names.extend(icon_col_names)

  return summary, is_flipped, feature_names


def biosppy_preprocessor(sig_i_np, Fs, sample_index, class_id, enabled):
  # filter ecg signal with biosspy first
  rpeaks_biosppy = np.nan
  no_rpeaks_biosppy = np.nan
  filtered_biosppy = np.nan

  if enabled:
    try:
      out = ecg.ecg(signal=sig_i_np, sampling_rate=Fs, show=False)
      
      # ts (array) – Signal time axis reference (seconds).
      # filtered (array) – Filtered ECG signal.
      # rpeaks (array) – R-peak location indices.
      # templates_ts (array) – Templates time axis reference (seconds).
      # templates (array) – Extracted heartbeat templates.
      # heart_rate_ts (array) – Heart rate time axis reference (seconds).
      # heart_rate (array) – Instantaneous heart rate (bpm).

      (ts, filtered_biosppy, rpeaks_biosppy, templates_ts, 
      templates, heart_rate_ts, heart_rate) = out
      
    except Exception:
      logging.info(f'biosppy crashed for sample {sample_index} in class {class_id}')

  return rpeaks_biosppy, filtered_biosppy


def global_signal_statistics(signals, df_analyze, rpeaks_biosppy):
  feature_names =  [ 
    'ECG_Quality_Mean', 
    'ECG_Quality_STD',
    'ECG_Rate_Mean', 
    'ECG_HRV',
    'R_P_biosppy' 
  ]

  # calculate the mean and standard devation of the signal quality
  ecg_q_mean = signals['ECG_Quality'].mean() 
  ecg_q_std = signals['ECG_Quality'].std()
  
  # consolidate the features for sample i
  #feat_i = [class_id] # not required and not known for X_test
  feat_i = [
    ecg_q_mean, # ECG_Quality_Mean
    ecg_q_std, # ECG_Quality_STD
    df_analyze.loc[0,'ECG_Rate_Mean'], # ECG_Rate_Mean
    df_analyze.loc[0,'HRV_RMSSD'], # ECG_HRV
    len(rpeaks_biosppy), #no. of detected r-peaks in biosspy (R_P_biosppy)
  ]
  # extend: Extends list by appending elements from the iterable.
  return feat_i, feature_names


# TODO: remove_outlier, ecg_quality_check
def process_signal(sig_i_np, y,  sample_index,
                  sampling_rate, 
                  remove_outlier, biosppy_enabled, ecg_quality_check, check_is_flipped):
  
  ######################### Set Flags & Remove NaN etc #########################
  # Preprocessing of ECG time series with Neurokit2 using customized function
  Fs = sampling_rate
  sample_id = sig_i_np[0] # sample_id, TODO: is amplitude, not sample ID!!

  # Remove NaN and \n from the array and cast to float64
  sig_i_np = sig_i_np.replace(to_replace=['NaN','\\n'],value=np.nan).dropna().to_numpy().astype('float64')
  raw_signal = sig_i_np.copy() # copy the raw signal for plotting

  # Extract the  y label for warnings etc.
  class_id = np.nan if y is None else y.iloc[sample_index].values[0] 

  # perform the data extraction / flipping / etc.
  rpeaks_biosppy, \
  filtered_biosppy, \
  signals, \
  peak_summary_neurokit, \
  default_feat_i, \
  filter_mask, \
  is_flipped, \
  feature_names = recursion(
      sig_i_np, Fs,
      sample_index,
      class_id,
      biosppy_enabled=biosppy_enabled,
      check_flipping=check_is_flipped
  )

  # the minimum of what we know if we crash later on
  feat_i = default_feat_i 

  # calculate ecg signal HR indicators if nk2 was successful
  qc_success = False
  if isinstance(signals, pd.DataFrame):
    try: 
      df_analyze = nk.ecg_analyze(signals, sampling_rate=Fs, method='auto')

      # TODO: regenerate peak summary before the statistics extraction
      # compute relevant statistics on filtered signals
      glob_feat_i, glob_feature_names = global_signal_statistics(
          signals, df_analyze, rpeaks_biosppy
        )

      glob_feat_i.extend(peak_summary_neurokit)
      glob_feature_names.extend(feature_names)      
      feature_names = glob_feature_names
      feat_i = glob_feat_i
    except AttributeError:
      logging.info(f'neurokit2-analyze crashed for sample {sample_index} in class {class_id}')
      filter_mask = np.ones(raw_signal.shape[0])

  ######################### organize data for export #########################
  plot_data = [
    sample_id, # np.int64
    class_id,  # np.int64
    raw_signal, # raw ecg  # ndarray
    rpeaks_biosppy, # ndarray
    filtered_biosppy, # ndarray
    signals, # signals_neurokit # DataFrame
    filter_mask,
    qc_success,
    is_flipped
  ]
  return feat_i, class_id, plot_data, feature_names

  #return (np.array(feat_i, dtype=float), class_id, plot_data)



def recursion(sig_i_np, Fs, sample_index, class_id, 
              biosppy_enabled, check_flipping=True):
  ######################### 1. biosppy trial #########################
  # Try raw biosppy preprocessing first before we try to use neurokit
  rpeaks_biosppy, filtered_biosppy = biosppy_preprocessor(
    sig_i_np,Fs, sample_index, class_id, enabled=biosppy_enabled)

  # feat_i config using just the biosppy information (minimal config)
  no_rpeaks_biosppy = len(rpeaks_biosppy)
  n = 110 + 20 # TODO FIX HARDCODE
  default_feat_i = [np.nan]*n
  default_feat_i[4] = no_rpeaks_biosppy #maybe biosppy worked
  signals = np.nan #initialize with False will check and flip signals
  peak_summary_neurokit = np.nan
  filter_mask = np.ones(sig_i_np.shape[0])
  feature_names = dist_df_colnames('.mean')
  feature_names.extend(dist_df_colnames('.std'))

  ######################### 1. neurokit trial #########################
  try: 
    with warnings.catch_warnings():
      # RuntimeWarning: Mean of empty slice.
      warnings.simplefilter("ignore", category=RuntimeWarning)
      signals, info = nk2_ecg_process_AML(sig_i_np, sampling_rate=Fs)

    # filter signals for peak counts, amplitudes, and QRS event duration
    filter_mask = np.ones(sig_i_np.shape[0], dtype=np.bool_)
    _ , is_flipped, _ = calc_peak_summary(
      signals=signals, sampling_rate=Fs, filter_mask=filter_mask
    )

    #repeat feature extraction with flipped signal
    # we only flip once (check_flipping) to avoid inf. loops
    if is_flipped and check_flipping:
      logging.info(f'swapped signal detected: mirroring sample {sample_index} in class {class_id}')
      #mirror the signal
      sig_i_np = -sig_i_np
      return recursion( 
          sig_i_np, Fs, sample_index, class_id,
          biosppy_enabled, check_flipping=False
        )

    # Filter and create summary after flipping
    # filter_mask = create_filter_mask(sig_i_np, signals, Fs)

    # filter signals for peak counts, amplitudes, and QRS event duration
    peak_summary_neurokit, _, feature_names = calc_peak_summary(
      signals=signals, sampling_rate=Fs, filter_mask=filter_mask
    )



  except (ValueError, IndexError, AttributeError):
    if check_flipping == False:
      logging.info(f'neurokit crashed after flipping sample {sample_index} in class {class_id}')
    else:
      logging.info(f'neurokit2 crashed for sample {sample_index} in class {class_id}')  
    signals = np.nan
    peak_summary_neurokit = np.nan


  return rpeaks_biosppy, \
    filtered_biosppy, \
    signals, \
    peak_summary_neurokit, \
    default_feat_i, \
    filter_mask, \
    not check_flipping, \
    feature_names


# Extract features from ECGs
def extract_features(run_cfg, env_cfg, df, y=None, verbose=False):

  # short_df_len = 10
  # df = df.iloc[0:short_df_len]
  # if isinstance(y, pd.DataFrame):
  #   y = y.iloc[0:short_df_len]

  # Predefine important variables
  Fs = run_cfg['sampling_rate']
  remove_outlier=run_cfg['preproc/remove_outlier/enabled']
  biosppy_enabled=run_cfg['preproc/filtering_biosppy/enabled']
  ecg_quality_check=run_cfg['preproc/ecg_quality_check/enabled']
  ecg_quality_threshold=run_cfg['preproc/ecg_quality_threshold']
  check_is_flipped=run_cfg['preproc/check_is_flipped/enabled']

  if remove_outlier:
    logging.info('Removing ecg outliers with pyheart... NOT IMPLEMENTED YET!')
      
  if biosppy_enabled:
    logging.info('Filtering with biosspy activated.')
  
  # for all the rows in the df
  results = Parallel(n_jobs=env_cfg['n_jobs'])(
    delayed(process_signal)
      (
        df.iloc[i, :],
        y, 
        i, # sample index
        Fs, 
        remove_outlier, biosppy_enabled, ecg_quality_check, check_is_flipped # flags
      )
      
      for i in tqdm(range(len(df))))

  # res is a touple (features, class_id)
  no_nan_mask =  [np.sum(np.isnan(res[0][0:14])) == 0 for res in results]

  # take the feature list from the first sample because its  length is constant
  feature_list = results[0][3]
  # Define F array to aggregate extracted sample features
  F=np.zeros([df.shape[0],len(feature_list)])

  # Define PD as a list array to aggregate extracted sample infos (for later plotting)
  # PD columns: [0:sample id | 1: class id | 2: raw signal| 3: r_peaks_biosspy | 4: filtered biosppy | 5: signals neurokit ]
  # PD rows: number of ecg signals
  plotData = np.zeros(shape=(df.shape[0], len(results[0][2])), dtype=np.object)

  for i, res in enumerate(results):
    if no_nan_mask[i] == True:
      F[i,:] = res[0]
      plotData[i,:] = res[2]

  # TODO: maybe plot the failing ones as well
  feat_df = pd.DataFrame(data=F,columns=feature_list)
  # profile = ProfileReport(feat_df, title="Pandas Profiling Report", explorative=True)
  # profile.to_file(Path("feature_extraction_report.html"))
  n_failures = np.sum(np.logical_not(no_nan_mask))
  logging.warning(f'features of {n_failures} samples could not be extracted')
  logging.warning(f'{len(no_nan_mask)-df.shape[0]} samples lost')
  # app = create_app(plotData)
  # app.run_server(debug=False)
  return(feat_df, y, plotData, no_nan_mask)

# Imports
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from biosppy.signals import ecg
#from ecgdetectors import Detectors
#from hrv import HRV
import neurokit2 as nk

from joblib import Parallel, delayed  
from tqdm import tqdm  
from preproc_viewer import create_app

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
def ecg_process_AML(ecg_signal, sampling_rate=300):
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
def calc_peak_summary(is_flipped, signals, sampling_rate):
  
  # Peak summary
  summary = []
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
    summary.append(p_mean)
    p_std = sig_pp['ECG_Clean'].std()
    summary.append(p_std)
  else:
    p_mean = np.nan
    p_std = np.nan
  
  # Peak s amplitude
  if len(sig_ss) > 0:
    s_mean = sig_ss['ECG_Clean'].mean()
    summary.append(s_mean)
    s_std = sig_ss['ECG_Clean'].std()
    summary.append(s_std)
  else:
    s_mean = np.nan
    s_std = np.nan
  
   #check whether the signal is flipped
  if not is_flipped and (
    len(sig_qq) > 0 and len(sig_rr) > 0 and
    abs(sig_qq['ECG_Clean'].mean()) > abs(sig_rr['ECG_Clean'].mean()) or
    len(sig_ss) > 0 and len(sig_rr) > 0 and
    abs(sig_ss['ECG_Clean'].mean()) > abs(sig_rr['ECG_Clean'].mean())
  ):
    is_flipped = True
    return summary, is_flipped #exit the function and try again with flipped signal
  
  # QRS duration
  sig_r_onset = signals[signals['ECG_R_Onsets'] == 1]
  sig_r_offset = signals[signals['ECG_R_Offsets'] == 1]
  if (len(sig_r_onset) == len(sig_r_offset)):
    d_qrs_N = sig_r_offset.index.to_numpy().ravel() - sig_r_onset.index.to_numpy().ravel() #number of samples between R Onset and Offset
    d_qrs_t = d_qrs_N / sampling_rate
    d_qrs_t_mean = d_qrs_t.mean()
    d_qrs_t_std = d_qrs_t.std()
  else:
    #TODO: in case of unenven R Onset and Offset detection develop more sofisticated algo to check which peaks can be retained?
    d_qrs_t_mean = np.nan
    d_qrs_t_std = np.nan
      
  
  summary.append(d_qrs_t_mean)
  summary.append(d_qrs_t_std)
  
  return summary, is_flipped


def process_signal(sig_i_np, y,  sample_index,
                  sampling_rate, feature_list,
                  remove_outlier, biosppy_enabled, ecg_quality_check, check_is_flipped):
  Fs = sampling_rate
  sample_id = sig_i_np[0] # sample_id, TODO: is amplitude, not sample ID!!
  sig_i_np = sig_i_np.replace(to_replace=['NaN','\\n'],value=np.nan).dropna().to_numpy().astype('float64')
  raw_signal = sig_i_np.copy()

  if y is not None:
    class_id = y.iloc[sample_index].values[0] # Very convoluted way to get just the class integer
  else:
    class_id = np.nan

    
  # remove outliers using pyheart?
  if remove_outlier:
    dummy=1 #TODO: remove outliers using pyheart?
      
  
  # filter ecg signal with biosspy first
  if biosppy_enabled:
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
      
      no_rpeaks_biosppy = len(rpeaks_biosppy)

    except Exception:
      logging.info(f'biosppy crashed for sample {sample_index} in class {class_id}')
      rpeaks_biosppy = np.nan
      no_rpeaks_biosppy = np.nan
      filtered_biosppy = np.nan
  else:
    rpeaks_biosppy = np.nan
    no_rpeaks_biosppy = np.nan
    filtered_biosppy = np.nan

      
  # Preprocessing of ECG time series with Neurokit2 using customized function
  nk2_crash_after_sig_flip = False
  is_flipped = not check_is_flipped #initialize with False will check and flip signals
  try: 
    signals, info = ecg_process_AML(sig_i_np, sampling_rate=Fs)
    
    if ecg_quality_check:
        #TODO: keep only the signals with ecq quality above threshold?
        dummy=1
    
    # filter signals for peak counts, amplitudes, and QRS event duration
    peak_summary_neurokit, is_flipped = calc_peak_summary(is_flipped, signals=signals, sampling_rate=Fs)
    
    #repeat feature extraction with flipped signal
    if is_flipped:
      logging.info(f'swapped signal detected: mirroring sample {sample_index} in class {class_id}')
      
      #mirror the signal
      sig_i_np = -sig_i_np
      
      #biosppy
      rpeaks_biosppy = np.nan
      no_rpeaks_biosppy = np.nan
      filtered_biosppy = np.nan
      if biosppy_enabled:
        try:
          out = ecg.ecg(signal=sig_i_np, sampling_rate=Fs, show=False)
          
          (ts, filtered_biosppy, rpeaks_biosppy, templates_ts, 
          templates, heart_rate_ts, heart_rate) = out
          no_rpeaks_biosppy = len(rpeaks_biosppy)
        except Exception:
          logging.info(f'biosppy crashed after flipping sample {sample_index} in class {class_id}')
      
      #neurokit2
      try:
        signals, info = ecg_process_AML(sig_i_np, sampling_rate=Fs)
        peak_summary_neurokit, is_flipped = calc_peak_summary(is_flipped, signals=signals, sampling_rate=Fs)
      except Exception:
        logging.info(f'neurokit crashed after flipping sample {sample_index} in class {class_id}')
        nk2_crash_after_sig_flip = True
        
        n = len(feature_list)
        feat_i = [np.nan]*n
        feat_i[0] =  sample_id
        feat_i[5] = no_rpeaks_biosppy #maybe biosppy worked
        signals = np.nan
  
    # calculate ecg signal HR indicators
    if not nk2_crash_after_sig_flip:
      df_analyze = nk.ecg_analyze(signals, sampling_rate=Fs, method='auto')
      
      # calculate the mean and standard devation of the signal quality
      ecg_q_mean = signals['ECG_Quality'].mean() 
      ecg_q_std = signals['ECG_Quality'].std()
      
      # consolidate the features for sample i
      #feat_i = [class_id] # not required and not known for X_test
      feat_i = [ecg_q_mean] # ECG_Quality_Mean
      feat_i.append(ecg_q_std,) # ECG_Quality_STD
      feat_i.append(df_analyze.loc[0,'ECG_Rate_Mean']) # ECG_Rate_Mean
      feat_i.append(df_analyze.loc[0,'HRV_RMSSD']) # ECG_HRV
      feat_i.append(len(rpeaks_biosppy)) #no. of detected r-peaks in biosspy (R_P_biosppy)
      feat_i.extend(peak_summary_neurokit) # extend() does inplace update to the original list
  except Exception:
    logging.info(f'neurokit2 crashed for sample {sample_index} in class {class_id}')
    n = len(feature_list)
    feat_i = [np.nan]*n
    feat_i[0] =  sample_id
    feat_i[5] = no_rpeaks_biosppy #maybe biosppy worked
    signals = np.nan
  
  plot_data = [
    sample_id, # np.int64
    class_id,  # np.int64
    raw_signal, # raw ecg  # ndarray
    rpeaks_biosppy, # ndarray
    filtered_biosppy, # ndarray
    signals # signals_neurokit # DataFrame
  ]

  return (np.array(feat_i, dtype=float), class_id, plot_data)


# Extract features from ECGs
def extract_features(run_cfg, env_cfg, df, feature_list, y=None, verbose=False):

  # df = df.iloc[0:100]
  # y = y.iloc[0:100]

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
        Fs, feature_list, 
        remove_outlier, biosppy_enabled, ecg_quality_check, check_is_flipped # flags
      )
      #for i in tqdm(range(len(df))))
      for i in tqdm(range(400))) 

  # res is a touple (features, class_id)
  no_nan_mask =  [np.sum(np.isnan(res[0][0:14])) == 0 for res in results]

  # Define F array to aggregate extracted sample features
  F=np.zeros([df.shape[0],len(feature_list)])
  F=np.zeros([400,len(feature_list)])

  # Define PD as a list array to aggregate extracted sample infos (for later plotting)
  # PD columns: [0:sample id | 1: class id | 2: raw signal| 3: r_peaks_biosspy | 4: filtered biosppy | 5: signals neurokit ]
  # PD rows: number of ecg signals
  plotData = np.zeros(shape=(df.shape[0], 6), dtype=np.object)

  for i, res in enumerate(results):
    if no_nan_mask[i] == True:
      F[i,:] = res[0]
      plotData[i,:] = res[2]

  feat_df = pd.DataFrame(data=F,columns=feature_list)
  feat_df = feat_df[no_nan_mask]
  
  # with .predict method y is set to None
  if isinstance(y, pd.DataFrame):
    y = y.iloc[0:400]
    y = y[no_nan_mask]
     
  # app = create_app(plotData)
  # app.run_server(debug=False)
  return(feat_df, y, plotData)
#%% Imports
# from .modules import ConfigLoader
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

# Load configs
# env_cfg = ConfigLoader().from_file('env/env.yml')

#%% Populate container for plot signals
def populate_PlotData(PD, i, sample_id, class_id, raw_ecg, rpeaks_biosppy, filtered_biosppy , signals_neurokit):
    PD[i][0] = sample_id
    PD[i][1] = class_id
    PD[i][2] = raw_ecg
    PD[i][3] = rpeaks_biosppy
    PD[i][4] = filtered_biosppy
    PD[i][5] = signals_neurokit

    return PD

#%%Load Data Set
def load_data(repopath):
    X = pd.read_csv(os.path.join(repopath, 'project3_ines', 'X_train_small.csv'))
    y = pd.read_csv(os.path.join(repopath, 'project3_ines', 'y_train_small.csv'))
    #X_test = pd.read_csv(os.path.join(env_cfg["datasets/project2/path"],'X_test.csv'))
    X_test = 0
    logging.info('Dataset imported')
    
    return (X, y, X_test)

#%%Split Classes
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

##% Define more flexible ecg_process function
def ecg_process_AML(ecg_signal, sampling_rate=300, method="neurokit"): #TODO method not used
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
      
#%% Extracted peaks summary
def calc_peak_summary(signals, sampling_rate):
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
    p_mean = sig_pp['ECG_Clean'].mean()
    summary.append(p_mean)
    p_std = sig_pp['ECG_Clean'].std()
    summary.append(p_std)
    
    # Peak s amplitude
    s_mean = sig_ss['ECG_Clean'].mean()
    summary.append(s_mean)
    s_std = sig_ss['ECG_Clean'].std()
    summary.append(s_std)
    
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
    
    return summary

#%% extract features from ECGs
def extract_features(df, Fs, feature_list, remove_outlier, biosppy_enabled, ecg_quality_check, ecg_quality_threshold, class_id, verbose):
    
    if remove_outlier:
        logging.info('Removing ecg outliers with pyheart...')
        
    if biosppy_enabled:
        logging.info('Filtering with biosspy activated.')
    
    # Define F array to aggregate extracted sample features
    F=np.zeros([df.shape[0],len(feature_list)])
    
    # Define PD as a list array to aggregate extracted sample infos (for later plotting)
    # PD columns: [0:sample id | 1: class id | 2: raw signal| 3: r_peaks_biosspy | 4: filtered biosppy | 5: signals neurokit ]
    # PD rows: number of ecg signals
    plotData = []
    for n_row in range(df.shape[0]):
        column = []
        for n_col in range(6):
            column.append(0)
            plotData.append(column)
    
    # for all the rows in the df
    for i in range(len(df)):
        sig_i_np = df.iloc[i, 1:].dropna().to_numpy()
        
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
                logging.info(f'biosppy crashed for sample {i} in class {class_id}')
                rpeaks_biosppy = np.nan
                no_rpeaks_biosppy = np.nan
                filtered_biosppy = np.nan
        else:
            rpeaks_biosppy = np.nan
            no_rpeaks_biosppy = np.nan
            filtered_biosppy = np.nan

            
        # Preprocessing of ECG time series with Neurokit2 using customized function
        try: 
            signals, info = ecg_process_AML(sig_i_np, sampling_rate=Fs)
            
            if ecg_quality_check:
                #TODO: keep only the signals with ecq quality above threshold?
                dummy=1
            
            # calculate ecg signal HR indicators
            df_analyze = nk.ecg_analyze(signals, sampling_rate=Fs, method='auto')
            
            # filter signals for peak counts, amplitudes, and QRS event duration
            peak_summary_neurokit = calc_peak_summary(signals=signals, sampling_rate=Fs)
            
            # calculate the mean and standard devation of the signal quality
            ecg_q_mean = signals['ECG_Quality'].mean() 
            ecg_q_std = signals['ECG_Quality'].std()
            
            # consolidate the features for sample i
            feat_i = [df.iloc[i,0]] # init a list with sample id
            feat_i.append(ecg_q_mean)
            feat_i.append(ecg_q_std,)
            feat_i.append(df_analyze.loc[0,'ECG_Rate_Mean'])
            feat_i.append(df_analyze.loc[0,'HRV_RMSSD'])
            feat_i.append(len(rpeaks_biosppy)) #no. of detected r-peaks in biosspy
            feat_i.extend(peak_summary_neurokit) # extend() does inplace update to the original list
        except Exception:
            logging.info(f'neurokit2 crashed for sample {i} in class {class_id}')
            n = len(feature_list)
            feat_i = [np.nan]*n
            feat_i[0] = df.iloc[i,0] # sample id
            feat_i[5] = no_rpeaks_biosppy #maybe biosppy worked
        
        F[i,:] = feat_i
        plotData = populate_PlotData(plotData,i,df.iloc[i,0],class_id,sig_i_np,rpeaks_biosppy,filtered_biosppy,signals)
        if verbose:
            sample_left = df.shape[0]-i
            print(f'Preprocessed ECG sample {i}({df.iloc[i,0]}) in class {class_id}... {sample_left} samples to go!')
        #TODO: in a suitable container collect the sample id and the signals dataframe (output of neurokit), which
        #which contains all the info for the plots
    
    feat_df = pd.DataFrame(data=F,columns=feature_list)
    
    return(feat_df, plotData)    
     
#%% Main

repopath = '/Users/inespereira/Documents/Github/aml2020'
os.chdir(repopath)
SAMPLING_RATE = 300

#%% Load data from repo (keep sample id for later use)
X, y, X_test = load_data(repopath)

#%% Split the original dataframe according to class
X0, X1, X2, X3 = split_classes(X, y)

#%% Define dataframe template in which will be filled with the extracted features
feature_list = ['Sample_Id', 
                'ECQ_Quality_Mean', 'ECQ_Quality_STD', 
                'ECG_Rate_Mean', 'ECG_Rate_STD',
                'R_P_biosppy', 'P_P/R_P', 'Q_P/R_P', 'R_P_neurokit' ,'S_P/R_P', 'T_P/R_P',  #relative number of peaks TODO
                'P_Amp_Mean', 'P_Amp_STD', 'S_Amp_Mean', 'S_Amp_STD',
                'QRS_t_Mean', 'QRS_t_STD']


#%% Feature extraction class 0
X0_features, X0_plotData = extract_features(df=X0,
                               Fs = SAMPLING_RATE,
                               feature_list = feature_list, 
                               remove_outlier=True, 
                               biosppy_enabled=True, 
                               ecg_quality_check=True, 
                               ecg_quality_threshold=0.8, 
                               class_id=0,
                               verbose=True
                               )

X0_features.head()
#%% Feature extraction class 1
X1_features, X1_plotData = extract_features(df=X1,
                               Fs = SAMPLING_RATE,
                               feature_list = feature_list, 
                               remove_outlier=False, 
                               biosppy_enabled=True, 
                               ecg_quality_check=False, 
                               ecg_quality_threshold=0.8, 
                               class_id=1,
                               verbose=True
                               )
X1_features.head()
#%% Feature extraction class 2
X2_features, X2_plotData = extract_features(df=X2,
                               Fs = SAMPLING_RATE,
                               feature_list = feature_list, 
                               remove_outlier=True, 
                               biosppy_enabled=True, 
                               ecg_quality_check=True, 
                               ecg_quality_threshold=0.8, 
                               class_id=2,
                               verbose=True
                               )
X2_features.head()
#%% Write pickle or similar
#TODO: write to pickle or similar for the features dataframes and ecg prepocessing
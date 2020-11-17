#%% Imports
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import os
repopath = '/Users/francescofusaro/Documents/Github/aml2020'
os.chdir(repopath)

from biosppy.signals import ecg
#from ecgdetectors import Detectors
#from hrv import HRV
import neurokit2 as nk

#%%Load Data Set
def load_data(repopath):
    X = pd.read_csv(f'{repopath}/project3_ffu/X_train.csv')
    y = pd.read_csv(f'{repopath}/project3_ffu/y_train.csv')
    X_test = pd.read_csv(f'{repopath}/project3_ffu/X_test.csv')
    
    logging.info('Dataset imported')
    
    return (X, y, X_test)

#%%Split Classes
def split_classes(X,y):
    class0_ls = y.index[y['y'] == 0].tolist() #healthy
    class1_ls = y.index[y['y'] == 1].tolist() #Arrhythmia1
    class2_ls = y.index[y['y'] == 2].tolist() #Arrhythmia2
    class3_ls = y.index[y['y'] == 3].tolist() #Noise
    
    X0 = X.iloc[class0_ls,:]
    df_X0 = pd.DataFrame(data=X0,columns=X.columns)
    
    X1 = X.iloc[class1_ls,:]
    df_X1 = pd.DataFrame(data=X1,columns=X.columns)
    
    X2 = X.iloc[class2_ls,:]
    df_X2 = pd.DataFrame(data=X2,columns=X.columns)
    
    X3 = X.iloc[class3_ls,:]
    df_X3 = pd.DataFrame(data=X3,columns=X.columns)
    
    return(df_X0, df_X1, df_X2, df_X3)

#%% Extracted peaks summary
def calc_peak_summary(signals, sampling_rate):
    #peak summary
    summary = []
    sig_qq = signals[signals['ECG_Q_Peaks'] == 1]
    q_count = len(sig_qq)
    sig_rr = signals[signals['ECG_R_Peaks'] == 1]
    r_count = len(sig_rr)
    sig_pp = signals[signals['ECG_P_Peaks'] == 1]
    p_count= len(sig_pp)
    sig_ss = signals[signals['ECG_S_Peaks'] == 1]
    s_count = len(sig_ss)
    sig_tt = signals[signals['ECG_T_Peaks'] == 1]
    t_count = len(sig_tt)
    
    #peak counts
    p_rel = p_count/r_count
    q_rel = q_count/r_count
    s_rel = s_count/r_count
    t_rel = t_count/r_count
    summary.append(p_rel)
    summary.append(q_rel)
    summary.append(r_count)
    summary.append(s_rel)
    summary.append(t_rel)
    
    #peak p amplitude
    p_mean = sig_pp['ECG_Clean'].mean()
    summary.append(p_mean)
    p_std = sig_pp['ECG_Clean'].std()
    summary.append(p_std)
    
    #peak s amplitude
    s_mean = sig_ss['ECG_Clean'].mean()
    summary.append(s_mean)
    s_std = sig_ss['ECG_Clean'].std()
    summary.append(s_std)
    
    #QRS duration
    d_qrs_N = sig_ss.index.to_numpy().ravel() - sig_qq.index.to_numpy().ravel() #number of samples between Q and R
    d_qrs_t = (d_qrs_N - 1) / sampling_rate
    d_qrs_t_mean = d_qrs_t.mean()
    d_qrs_t_std = d_qrs_t.std()
    
    summary.append(d_qrs_t_mean)
    summary.append(d_qrs_t_std)
    
    return summary

#%% extract features from ECGs
def extract_features(df, Fs, feature_list, remove_outlier, biosspy_cleaning, ecg_quality_check, ecg_quality_threshold, class_id):
    
    if remove_outlier:
        logging.info('Removing ecg outliers with pyheart...')
        
    if biosspy_cleaning:
        logging.info('Pre-filtering ECG with biosspy')
    
    # Define F array to aggregate extracted sample features
    F=np.zeros([df.shape[0],len(feature_list)])
    
    # for all the rows in the df
    for i in range(len(df)):
        sig_i = df.iloc[i,1:] #signal i wo sample id
        sig_i = sig_i.replace(to_replace='NaN',value=np.nan)
        sig_i_np = (sig_i.to_numpy()).ravel()
        sig_i_np = sig_i_np[~np.isnan(sig_i_np)]
        
        # remove outliers using pyheart?
        if remove_outlier:
            dummy=1 #TODO: remove outliers using pyheart
            
        
        # filter ecg signal with biosspy first
        if biosspy_cleaning:
            out = ecg.ecg(signal=sig_i_np, sampling_rate=Fs, show=False)
            
            # ts (array) – Signal time axis reference (seconds).
            # filtered (array) – Filtered ECG signal.
            # rpeaks (array) – R-peak location indices.
            # templates_ts (array) – Templates time axis reference (seconds).
            # templates (array) – Extracted heartbeat templates.
            # heart_rate_ts (array) – Heart rate time axis reference (seconds).
            # heart_rate (array) – Instantaneous heart rate (bpm).

            (ts, filtered, rpeaks, templates_ts, 
            templates, heart_rate_ts, heart_rate) = out
            
            sig_i_np = filtered
            
        # process ecg sample with with neurokit
        signals, info = nk.ecg_process(sig_i_np, sampling_rate=Fs)
        
        if ecg_quality_check:
            #TODO: keep only the signals with ecq quality above threshold
            dummy=1
        
        # calculate ecg signal HR indicators
        df_analyze = nk.ecg_analyze(signals, sampling_rate=Fs, method='auto')
        
        # filter signal for peak counts, amplitudes, and QRS event duration
        peak_summary_neurokit = calc_peak_summary(signals=signals, sampling_rate=Fs)
        
        # calculate the mean and standard devation of the signal quality
        ecg_q_mean = signals['ECG_Quality'].mean() #TODO check naming
        ecg_q_std = signals['ECG_Quality'].std()   #TODO check naming
        
        # consolidate the features for sample i
        feat_i = [df.iloc[i,0]] # sample id
        feat_i.append(ecg_q_mean)
        feat_i.append(ecg_q_std,)
        feat_i.append(df_analyze.iloc[0,0]) #ECG_Rate_Mean
        feat_i.append(df_analyze.iloc[0,1]) #HRV_RMSSD
        feat_i.append(len(rpeaks)) #no. of detected r-peaks in biosspy
        for elem in peak_summary_neurokit:
            feat_i.append(elem)
        
        #TODO aggregate feat_i into F_array
        F[i,:] = feat_i
        
    #TODO build a dataframe with aggregated feat_i
    
    feat_df = pd.DataFrame(data=F,columns=feature_list)
    
    return(feat_df)    
     

#%% Main

repopath = '/Users/francescofusaro/Documents/Github/aml2020'
os.chdir(repopath)

#%% Load data from repo (keep sample id for later use)
X, y, X_test = load_data(repopath)

#%% Split the original dataframe according to class
X0, X1, X2, X3 = split_classes(X, y)

#%% Define dataframe template in which will be filled with the extracted features
feature_list = ['Sample_Id', 
                'ECQ_Quality_Mean', 'ECQ_Quality_STD', 
                'ECG_Rate_Mean', 'ECG_Rate_STD'
                'R_P_biosppy', 'P_P/R_P', 'Q_P/R_P', 'R_P_neurokit' ,'S_P/R_P', 'T_P/R_P',  #relative number of peaks TODO
                'P_Amp_Mean', 'P_Amp_STD', 'S_Amp_Mean', 'S_Amp_STD',
                'QRS_t_Mean', 'QRS_t_STD']


#%% Feature extraction class 0
X0_features = extract_features(df=X0,
                               Fs = 300,
                               feature_list = feature_list, 
                               remove_outlier=True, 
                               biosspy_cleaning=True, 
                               ecg_quality_check=True, 
                               ecg_quality_threshold=0.8, 
                               class_id='0')

X0_features.head()
#%%
print(X.shape[1])
F=np.zeros([X0.shape[0],len(feature_list)])
print(F.shape)
# %%

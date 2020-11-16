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

def calc_peak_summary(df_tot, sampling_rate):
    
    return peak_summary

#%% extract features from ECGs
def extract_features(df, Fs, df_temp, remove_outlier, biosspy_cleaning, ecg_quality_check, ecg_quality_threshold, class_id):
    # replace 'NaN' strings
    df.replace(to_replace='NaN',value=np.nan,inplace=True)
    
    if remove_outlier:
        logging.info('Removing ecg outliers with pyheart...')
        
    if biosspy_cleaning:
        logging.info('Pre-filtering ECG with biosspy')
    
    # for all the rows in the df
    for i in range(len(df)):
        sig_i = df.iloc[i,1:] #signal i wo sample id
        sig_i_np = (sig_i.to_numpy()).ravel()
        sig_i_np = sig_i_np[~np.isnan(sig_i_np)]
        
        # remove outliers using pyheart?
        if remove_outlier:
            x=1 #TODO: remove outliers using pyheart
            
        
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
            x=1
        
        # calculate ecg signal HR indicators
        df_analyze = nk.ecg_analyze(signals, sampling_rate=Fs, method='auto')
        
        # filter signal for peak counts, amplitudes, and QRS event duration
        peak_summary = calc_peak_summary(df_tot=sig, sampling_rate=Fs)
        
        
        
    
    return(df_temp)    
     

#%% Main

repopath = '/Users/francescofusaro/Documents/Github/aml2020'
os.chdir(repopath)

#%% Load data from repo (keep sample id for later use)
X, y, X_test = load_data(repopath)

#%% Split the original dataframe according to class
X0, X1, X2, X3 = split_classes(X, y)

#%% Define dataframe template in which will be filled with the extracted features
df_template = pd.DataFrame(columns=['Sample_Id', 
                                    'ECQ_Quality', 'ECG_Rate_Mean',
                                    'P_P/R_P', 'Q_P/R_P', 'R_P', 'S_R', 'T_R', #relative number of peaks
                                    'R_Amp_Mean', 'R_AMP_SD'])


#%% Feature extraction class 0
X0_features = extract_features(df=X0,
                               Fs = 300,
                               df_temp = df_template, 
                               remove_outlier=True, 
                               biosspy_cleaning=True, 
                               ecg_quality_check=True, 
                               ecg_quality_threshold=0.8, 
                               class_id='0')
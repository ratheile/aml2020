import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import logging
import os.path

def feature_reduction(df_x,cvmin):
#identify columns which contains constant values and some noise.
#use the coefficient of variation CV as metric (CV = sigma/mean).
#drop columns in which CV < cvmin.
 dropped_cols_mean = [] #zero mean
 dropped_cols_cv = [] #CV
 n = 0
 n_tot = len(df_x.columns)
 for column in df_x:
     mean = df_x[column].astype("float").mean()
     if mean == 0:
         dropped_cols_mean.append(column)
         n = n+1
     else: 
         cv = df_x[column].astype("float").std()/mean
         if cv < cvmin:
             dropped_cols_cv.append(column)
             n = n+1
 print("dropped",n,"out of",n_tot)
 #drop the columns with mean = 0 and cv < cv_min
 df_x.drop(dropped_cols_mean,axis=1,inplace=True)
 df_x.drop(dropped_cols_cv,axis=1,inplace=True)
 return(df_x)

def remove_nan(df_x,tool):
#replace nans    
 if tool == "mean":    
     for column in df_x:
         if df_x[column].isnull().values.any():
             mean = df_x[column].astype("float").mean()
             df_x[column].replace(np.nan, mean, inplace=True)
 elif tool == "median":
     for column in df_x:
         if df_x[column].isnull().values.any():
             mean = df_x[column].astype("float").median()
             df_x[column].replace(np.nan, mean, inplace=True)
 elif tool == "freq":
     for column in df_x:
         if df_x[column].isnull().values.any():
             mean = df_x[column].astype("float").mode()
             df_x[column].replace(np.nan, mean, inplace=True)
 return(df_x)

def extract_y_correlated(df_x,df_y):
#calculate correlation with y and extract most correlated ones    
 
 #merge dataframes
 df = pd.concat([df_y,df_x],axis=1)
 df.drop(["id"],axis=1,inplace=True)
 
 #build the correlation matrix
 p = df.corr()
 
 #bin the correlation coefficients in three different classes
 p_y = p[["y"]]
 bins = np.linspace(-1, 1, 7)
 group_names = ["neg[1-0.67]", "neg[0.67-0.33]", "neg[0.33-0]", "[0-0.33]", "[0.33-0.67]", "[0.67-1]"]
 p_y.loc[:,"y-binned"] = pd.cut(p_y.loc[:,"y"], bins, labels=group_names, include_lowest=True)
 print("number of features by y-correlation bin:")
 print(p_y["y-binned"].value_counts())
 
 #features in top and bottom category
 p_y_033_067 = p_y[p_y["y-binned"]=="[0.33-0.67]"]
 p_y_n67_033 = p_y[p_y["y-binned"]=="neg[0.67-0.33]"]
 #l1=["y"]
 #l1.append(p_y_n67_033.index.tolist())
 l1=["y"]+p_y_033_067.index.tolist()+p_y_n67_033.index.tolist()
 print(l1)
 df2 = df[l1]
 print("columns kept after correlation check with y:", len(l1))
 return(df2)

def drop_x_correlated(df,lb):
#look at correlation between features

 #sort the features by correlation and define a lower bound for correlation
 p= df.corr()
 c = p.abs()
 s = c.unstack()
 so = s.sort_values(kind="quicksort")
 so=so[so>=lb]
 #the upperbound is one by definition
 so=so[so<1]
 
 #remove duplicated entries and create a unique list
 sf = so.to_frame() # series to dataframe
 sff = sf.drop_duplicates()
 lc_list = sff.index.tolist()
 sff.head(5)
 
 lc_clean = [] #the elements are nested in touples of two. put them in a list
 for lc in lc_list:
    lc_clean.append(lc[0])
    lc_clean.append(lc[1])
 
 # unique elements in the list
 lc_unique = [] 
 for x in lc_clean:
    if x not in lc_unique:
        lc_unique.append(x)
 print("Dropping features with correlation higher than", lb,":")
 print(lc_unique)

 #drop values from dataframe
 df.drop(lc_unique,axis=1,inplace=True)
 return(df)

def remove_outlier(df,cont_lim):
#use Isolation Forest to indentify outliers on the selected features
    iso_Y_arr, iso_X_arr, df_colnames = df_to_array(df)
    len_Y_outl = len(iso_Y_arr)
    
    iso = IsolationForest(contamination=cont_lim)
    yhat = iso.fit_predict(iso_X_arr)
    mask = yhat != -1
    iso_X_arr, iso_Y_arr = iso_X_arr[mask, :], iso_Y_arr[mask]

    len_Y_inl = len_Y_outl-sum(mask)
    print(f"Outliers removal:")
    print(f"Removing {len_Y_inl} outliers out of {len_Y_outl} datapoints.")
    
    numrows = len(iso_X_arr)    
    numcols = len(iso_X_arr[0])
    iso_arr = np.random.rand(numrows,numcols+1)
    iso_arr[:,0]=iso_Y_arr
    iso_arr[:,1:(numcols+1)]=iso_X_arr
    df_iso = pd.DataFrame(data=iso_arr, columns=df_colnames)
    return(df_iso)

def apply_pca(df, n_comp):
    X_arr, Y_arr = normalize(df)

    pca = PCA(n_components=2)
    pca.fit(X_arr)
    X_pca=pca.transform(X_arr)
    print("\nPC 1 with scaling:\n", pca.components_[0])
    return(X_pca, Y_arr)

def normalize(df):
    Y_arr, X_arr, df_colnames = df_to_array(df)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_arr_scaled= min_max_scaler.fit_transform(X_arr)
    return(X_arr_scaled, Y_arr)

def df_to_array(df):
    Y_arr = df["y"].to_numpy()
    df_x = df.drop(["y"],axis=1)
    X_arr = df_x.to_numpy()
    df_colnames = df.columns.tolist()
    return(Y_arr,X_arr,df_colnames)

def run(run_cfg, env_cfg):
#main script
    logging.warn(env_cfg)

    #load the data
    #absolute path data
    path = '~/Documents/GitHub/aml2020/task1'

    #Load in the training data
    fname = 'X_train.csv'
    fpath = os.path.join(path,fname)
    df_x = pd.read_csv(fpath)

    fname = 'y_train.csv'
    fpath = os.path.join(path,fname)
    df_y = pd.read_csv(fpath)

    #remove features by coefficient of variance
    if run_cfg['preprocessing/feature_removal']:
     df_x = feature_reduction(df_x,run_cfg['preprocessing/feature_removal_cv_min'])

    #remove nan
    df_x = remove_nan(df_x,run_cfg['preprocessing/nan'])

    #extract features by y correlation
    if run_cfg['preprocessing/y_corr']:
     df_yx = extract_y_correlated(df_x,df_y)

    #remove features by x correlation
    if run_cfg['preprocessing/x_corr']:
     df_yx = drop_x_correlated(df_yx,run_cfg['preprocessing/x_corr_lb'])

    #remove outliers
    if run_cfg['preprocessing/outlier_removal']:
     df_yx = remove_outlier(df_yx,run_cfg['preprocessing/outlier_cont_lim'])

    #apply pca (with min max normalization)
    if run_cfg['preprocessing/apply_pca']:
     X, y = apply_pca(df_yx, run_cfg['preprocessing/pca_n_comp'])

    if run_cfg['preprocessing/apply_pca'] != 1:
        X,y = normalize(df_yx)
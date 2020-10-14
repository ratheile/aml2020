import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import logging
import os.path

def feature_reduction(df_x,cvmin,cvmax):
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
         if cv < cvmin or cv > cvmax:
             dropped_cols_cv.append(column)
             n = n+1
 print(f"dropped {n} out of {n_tot}")
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

def remove_outlier(df_x, df_y,cont_lim):
#use Isolation Forest to indentify outliers on the selected features
    #iso_Y_arr, iso_X_arr, df_colnames = df_to_array(df)
    iso_Y_arr = df_y.to_numpy()
    iso_X_arr = df_x.to_numpy()


    len_Y_outl = len(iso_Y_arr)
    
    iso = IsolationForest(contamination=cont_lim)
    yhat = iso.fit_predict(df_x)
    mask = yhat != -1
    iso_X_arr, iso_Y_arr = iso_X_arr[mask, :], iso_Y_arr[mask]

    len_Y_inl = len_Y_outl-sum(mask)
    print(f"Outliers removal:")
    print(f"Removing {len_Y_inl} outliers out of {len_Y_outl} datapoints.")
    
    #numrows = len(iso_X_arr)    
    #numcols = len(iso_X_arr[0])
    #iso_arr = np.random.rand(numrows,numcols+1)
    #iso_arr[:,0]=iso_Y_arr
    df_y_iso = pd.DataFrame(data=iso_Y_arr, columns=df_y.columns.tolist())
    df_x_iso = pd.DataFrame(data=iso_X_arr, columns=df_x.columns.tolist())
    return(df_x_iso, df_y_iso)

def extract_top_ETC(df_x,df_y,n_top,rnd_state):
# check feature importance with Extra Trees Classifier
# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e 
  model = ExtraTreesClassifier(random_state=rnd_state)
  print(df_x.head(3))
  print(df_y.head(3))
  model.fit(df_x,df_y)
  #print(model.feature_importances_)
  feat_importances = pd.Series(model.feature_importances_, index=df_x.columns)
  #print(feat_importances.nlargest(rnd_state).index)
  df_x_ETC = df_x[feat_importances.nlargest(rnd_state).index]
  print(df_x_ETC.head(5))
  return(df_x_ETC,df_y)

def extract_y_correlated(df_x,df_y):
#calculate correlation with y and extract most correlated ones    
 
 #merge dataframes
 df = pd.concat([df_y,df_x],axis=1)
 #df.drop(["id"],axis=1,inplace=True) #alreasy taken care of.
 
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
 l1=p_y_033_067.index.tolist()+p_y_n67_033.index.tolist()
 df_x_corr = df_x[l1]
 print(f"no. of columns kept after correlation check with y: {len(l1)}")
 print(f"columns kept after correlation check with y:\n {df_x_corr.columns.tolist()}")
 return(df_x,df_y)

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
 print(f"Dropping features with correlation higher than {lb}:")
 print(lc_unique)

 #drop values from dataframe
 df.drop(lc_unique,axis=1,inplace=True)
 print(f"These columns are retained:\n{df.columns.tolist()}")

 compare_to_bench(df.columns.tolist())
 return(df)

def apply_pca(df_x, df_y, n_comp):
    X_arr, Y_arr = normalize(df)

    pca = PCA(n_components=2)
    pca.fit(X_arr)
    X_pca=pca.transform(X_arr)
    print(f"PC 1 with scaling:\n {pca.components_[0]}")
    return(X_pca, Y_arr)

def normalize(df_x):
    Y_arr, X_arr, df_colnames = df_to_array(df_x)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_arr_scaled= min_max_scaler.fit_transform(X_arr)
    df_x = pd.DataFrame(data=X_arr_scaled,columns=df_colnames)
    return(X_arr_scaled, Y_arr)

def df_to_array(df_x):
    Y_arr = df_y.to_numpy()
    X_arr = df_x.to_numpy()
    df_x_colnames = df_x.columns.tolist()
    return(Y_arr,X_arr,df_colnames)

def compare_to_bench(cand_list):
  bench = ["x85", "x302", "x761", "x687", "x507", "x685", "x785", "x540", "x155", "x482", "x356", "x184", "x35", "x546", "x745", "x579", "x344", "x783", "x198", "x476"]
  lc_found = []
  lc_not_found =[]
  lc_not_bench = []
  for col in bench:
    if col in cand_list:
      lc_found.append(col)
    else:
      lc_not_found.append(col)

  for col in cand_list:
      if col not in bench:
        lc_not_bench.append(col)
  print(f"These bench columns were not found:\n{lc_not_found}")
  print(f"These bench columns were found:\n{lc_found}")
  print(f"These columns are not in bech:\n{lc_not_bench}")

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
    df_x.drop(["id"],axis=1,inplace=True)


    fname = 'y_train.csv'
    fpath = os.path.join(path,fname)
    df_y = pd.read_csv(fpath)
    df_y.drop(["id"],axis=1,inplace=True)

    print(df_x.head(3))
    print(df_y.head(3))

    #remove features by coefficient of variance
    if run_cfg['preprocessing/feature_removal']:
     df_x = feature_reduction(df_x,run_cfg['preprocessing/feature_removal_cv_min'],run_cfg['preprocessing/feature_removal_cv_max'])

    #remove nan
    df_x = remove_nan(df_x,run_cfg['preprocessing/nan'])

    #remove outliers
    if run_cfg['preprocessing/outlier_removal']:
     df_x, df_y = remove_outlier(df_x,df_y,run_cfg['preprocessing/outlier_cont_lim'])

    if run_cfg['preprocessing/extract_top_ETC']:
     df_x, df_y = extract_top_ETC(df_x,df_y,run_cfg['preprocessing/ETC_n_top'],run_cfg['preprocessing/ETC_rs'])
    
    #extract features by y correlation
    if run_cfg['preprocessing/y_corr']:
     df_yx = extract_y_correlated(df_x,df_y)

    #remove features by x correlation
    if run_cfg['preprocessing/x_corr']:
     df_x = drop_x_correlated(df_x,run_cfg['preprocessing/x_corr_lb'])

    #apply pca (with min max normalization)
    if run_cfg['preprocessing/apply_pca']:
     X, y = apply_pca(df_yx, run_cfg['preprocessing/pca_n_comp'])

    if run_cfg['preprocessing/apply_pca'] != 1:
        X,y = normalize(df_yx)
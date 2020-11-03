import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

def drop_feat_cov_constant(df_x, cvmin):
  # identify columns which contains constant values and some noise.
  # use the coefficient of variation CV as metric (CV = sigma/mean).
  # drop columns in which CV < cvmin.

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
  logging.info(f"COV - dropped {n} out of {n_tot}")
  #drop the columns with mean = 0 and cv < cv_min
  df_x.drop(dropped_cols_mean,axis=1,inplace=True)
  df_x.drop(dropped_cols_cv,axis=1,inplace=True)
  return(df_x)


def drop_feat_uncorrelated_y(df_x, df_y):
  #calculate correlation with y and extract most correlated ones    

  #merge dataframes
  df = pd.concat([df_y,df_x],axis=1)

  #build the correlation matrix
  p = df.corr()

  #bin the correlation coefficients in three different classes
  p_y = p[["y"]]
  bins = np.linspace(-1, 1, 7)
  group_names = ["neg[1-0.67]", "neg[0.67-0.33]", "neg[0.33-0]", "[0-0.33]", "[0.33-0.67]", "[0.67-1]"]
  p_y.loc[:,"y-binned"] = pd.cut(p_y.loc[:,"y"], bins, labels=group_names, include_lowest=True)
  logging.info("number of features by y-correlation bin:")
  logging.info(p_y["y-binned"].value_counts())

  #features in top and bottom category
  p_y_033_067 = p_y[p_y["y-binned"]=="[0.33-0.67]"]
  p_y_n67_033 = p_y[p_y["y-binned"]=="neg[0.67-0.33]"]
  #l1=["y"]
  #l1.append(p_y_n67_033.index.tolist())
  l1=p_y_033_067.index.tolist()+p_y_n67_033.index.tolist()
  logging.info(l1)
  df2 = df[l1]
  logging.info(f"columns kept after correlation check with y: {len(l1)}")
  return(df2)

def drop_feat_ETR(df_x,df_y,n_top,rnd_state,max_feat):
# check feature importance with Extra Trees Regressor
# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e 
  model = ExtraTreesRegressor(random_state=rnd_state,max_features=max_feat)
  model.fit(df_x,df_y)
  feat_importances = pd.Series(model.feature_importances_, index=df_x.columns)
  #print(feat_importances.nlargest(n_top))
  df_x_ETC = df_x[feat_importances.nlargest(n_top).index]
  #print(df_x_ETC.head(5))
  return(df_x_ETC)


def drop_feat_correlated_x(df, lb):#look at correlation between features
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
  logging.info(f"Dropping features with correlation higher than {lb}:")
  logging.info(lc_unique)

  #drop values from dataframe
  df.drop(lc_unique,axis=1,inplace=True)
  return(df)



def ffu_dim_reduction(run_cfg, X,y):
  # drop features by coefficient of variance
  if run_cfg['preproc/rmf/ffu/cov/enabled']:
    cvmin = run_cfg['preproc/rmf/ffu/cov/cvmin']
    X = drop_feat_cov_constant(X, cvmin) 

  # drop features by y correlation
  if run_cfg['preproc/rmf/ffu/y_corr/enabled']:
    X = drop_feat_uncorrelated_y(X, y)

  # drop features by Extra Tree Regressor
  if run_cfg['preproc/rmf/ffu/etr/enabled']:
    n_top = run_cfg['preproc/rmf/ffu/etr/n_top']
    rnd_state = run_cfg['preproc/rmf/ffu/etr/rnd_state']
    max_feat = run_cfg['preproc/rmf/ffu/etr/max_feat']
    X = drop_feat_ETR(X,y,n_top,rnd_state,max_feat)  

  # drop features by x correlation
  if run_cfg['preproc/rmf/ffu/x_corr/enabled']:
    lb = run_cfg['preproc/rmf/ffu/x_corr/lb']
    X = drop_feat_correlated_x(X,lb)

  return(X)
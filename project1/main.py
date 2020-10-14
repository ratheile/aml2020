import logging
import os
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import \
    RepeatedKFold, \
    cross_val_score, \
    train_test_split

from sklearn.linear_model import \
    LinearRegression, \
    Lasso, \
    Ridge, \
    ElasticNet

import lightgbm as lgbm
#from autofeat import FeatureSelector, AutoFeatRegressor

from scipy import stats



######################## Francesco #######################################################
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
  l1=["y"]+p_y_033_067.index.tolist()+p_y_n67_033.index.tolist()
  logging.info(l1)
  df2 = df[l1]
  logging.info(f"columns kept after correlation check with y: {len(l1)}")
  return(df2)


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

def remove_isolation_forest_outlier(df, cont_lim):
  #use Isolation Forest to indentify outliers on the selected features
  iso_Y_arr, iso_X_arr, df_colnames = df_to_array(df)
  len_Y_outl = len(iso_Y_arr)
  
  iso = IsolationForest(contamination=cont_lim)
  yhat = iso.fit_predict(iso_X_arr)
  mask = yhat != -1
  iso_X_arr, iso_Y_arr = iso_X_arr[mask, :], iso_Y_arr[mask]

  len_Y_inl = len_Y_outl-sum(mask)
  logging.info(f"Outliers removal:")
  logging.info(f"Removing {len_Y_inl} outliers out of {len_Y_outl} datapoints.")
  
  numrows = len(iso_X_arr)    
  numcols = len(iso_X_arr[0])
  iso_arr = np.random.rand(numrows,numcols+1)
  iso_arr[:,0]=iso_Y_arr
  iso_arr[:,1:(numcols+1)]=iso_X_arr
  df_iso = pd.DataFrame(data=iso_arr, columns=df_colnames)
  # TODO: has to return y
  return(df_iso)


def pca_dim_reduction(X, n_comp):
  pca = PCA(n_components=2)
  pca.fit(X)
  X_pca = pca.transform(X)
  logging.info(f"\nPC 1 with scaling:\n { pca.components_[0]}")
  return X


def normalize(X):
  min_max_scaler = MinMaxScaler()
  X_arr_scaled = min_max_scaler.fit_transform(X)
  return X


######################## Inês #######################################################


def find_isolation_forest_outlier(X,y,method, cont_lim):
  if method == 'isol_forest':
    clf = IsolationForest(contamination=cont_lim).fit(X)
    y_pred_train = clf.predict(X)
    # inliers = np.array(np.where(y_pred_train==1))
    outliers = np.array(np.where(y_pred_train==-1))
    # print(f"Here are a few examples of inliers:\n{inliers}")
    # print(f"The total number of inliers is: {inliers.size}")
    # print(f"Here are the detected outliers:\n{outliers}")
    print(f"The total number of outliers removed: {outliers.size}")
    outliers = outliers.tolist()
    outliers = [ item for elem in outliers for item in elem]
    X_inliers = X.drop(index=outliers)
    y_inliers = y.drop(index=outliers)
    return X_inliers, y_inliers

def rfe_dim_reduction(X,y,method):
  # Good read: https://scikit-learn.org/stable/modules/feature_selection.html
  # Also: https://www.datacamp.com/community/tutorials/feature-selection-python
  # Different types of feature selection methods:
  # 1. Filter methods: apply statistical measures to score features (corr coef and Chi^2).
  # 2. Wrapper methods: consider feature selection a search problem (e.g. RFE)
  # 3. Embedded methods: feature selection occurs with model training (e.g. LASSO)

  estimator = Ridge() # TODO: this is an arbitrary choice and the result is influenced by this!
  if method == "rfe":
    selector = RFE(estimator, n_features_to_select=60, step=10, verbose=0)
  elif method == "rfecv":
    selector = RFECV(estimator, step=1, cv=5, verbose=0, min_features_to_select=20)
  # TODO: consider other methods? e.g. tree-based feature selection + SelectFromModel?

  selector = selector.fit(X, y)
  print('Original number of features : %s' % X.shape[1])
  print("Final number of features : %d" % selector.n_features_)
  X_red = selector.transform(X)
  X_red = pd.DataFrame(X_red)

  return X_red
  
def autofeat_dim_reduction(X,y):
  fsel = FeatureSelector(verbose=1)
  X = fsel.fit_transform(X,y)
  return X

######################## Raffael #######################################################
def simple_fit(model, X, y):
  model = model.fit(X, y)
  return model 


def auto_crossval(model, X, y):
  rkf = RepeatedKFold(n_splits=10, n_repeats=2)

  scores = cross_val_score(
    model, X, y, cv=rkf, verbose=1,
    scoring='r2'
  )

  return scores

def cfg_to_estimators(run_cfg):
  elasticnet_cfg = run_cfg['models/elasticnet']
  lasso_cfg = run_cfg['models/lasso']
  ridge_cfg = run_cfg['models/ridge']
  estimators = {
    'elasticnet': {
      'model': lambda: ElasticNet(alpha=1.01),
      'fit': simple_fit,
      'crossval_fit': lambda m,X,y: auto_crossval(m,X,y),
      'validate': lambda m,X,y: m.score(m,X,y)
    },
    'lasso': {
      'model': lambda: Lasso(alpha=1.01),
      'fit': simple_fit,
      'crossval_fit': lambda m,X,y: auto_crossval(m,X,y),
      'validate': lambda m,X,y: m.score(m,X,y)
    },
    'ridge': {
      'model': lambda: Ridge(alpha=1.01),
      'fit': simple_fit,
      'crossval_fit': lambda m,X,y: auto_crossval(m,X,y),
      'validate': lambda m,X,y: m.score(m,X,y)
    },
    'lightgbm': {
      'model': lambda : lgbm.LGBMRegressor(),
      'fit': simple_fit,
      'crossval_fit': lambda m,X,y: auto_crossval(m,X,y),
      'validate': lambda m,X,y: m.score(m,X,y)
    }
  }
  return estimators

def pool_f(args):
  estimators = args['estimators']
  task_args = args['task_args']
  model_dict = estimators[task_args['task']]
  model = model_dict['model']() # factory
  crossval_fit = model_dict['crossval_fit']

  # run the task:
  X = task_args['X']
  y = task_args['y']
  return (crossval_fit(model, X, y), model)


def fill_nan(X, strategy):
  imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
  return(imputer.fit_transform(X))


def ffu_dim_reduction(run_cfg, X,y):
  # drop features by coefficient of variance
  if run_cfg['preproc/rmf/ffu/cov/enabled']:
    cvmin = run_cfg['preproc/rmf/ffu/cov/cvmin']
    X = drop_feat_cov_constant(X, cvmin) 

  # drop features by y correlation
  if run_cfg['preproc/rmf/ffu/y_corr/enabled']:
    Xy = drop_feat_uncorrelated_y(X, y)

  # drop features by x correlation
  if run_cfg['preproc/rmf/ffu/x_corr/enabled']:
    lb = run_cfg['preproc/rmf/ffu/x_corr/lb']
    Xy = drop_feat_correlated_x(Xy,lb)

  X = Xy.drop(["y"],axis=1)
  return(X)

def run(run_cfg, env_cfg):
  ###################### Preprocessing ##############################

  # Load training dataset from csv
  datapath = env_cfg['datasets/project1/path']
  X = pd.read_csv(f'{datapath}/X_train.csv')
  y = pd.read_csv(f'{datapath}/y_train.csv')
  X_u = pd.read_csv(f'{datapath}/X_test.csv') # unlabeled
  # remove index column
  y = y.iloc[:,1:]
  X = X.iloc[:,1:]
  logging.info('Training dataset imported')

  # remove NaN 
  X[:] = fill_nan(X, run_cfg['preproc/imputer/strategy'])

  # remove outliers (rows/datapoints)
  if run_cfg['preproc/outlier/enabled']:
    outlier_type = run_cfg['preproc/outlier/type']
    cont_lim =  run_cfg['preproc/outlier/cont_lim']
    if run_cfg['preproc/outlier/impl'] == 'ines':
      X,y = find_isolation_forest_outlier(X,y,outlier_type, cont_lim)
    else:
      X = remove_isolation_forest_outlier(X, cont_lim)

  rfe_method = run_cfg['preproc/rmf/rfe/method']
  rmf_pipelines = {
    'ffu': lambda X,y: ffu_dim_reduction(run_cfg,X,y),
    'rfe': lambda X,y: rfe_dim_reduction(X,y,rfe_method),
    'auto' : lambda X,y: autofeat_dim_reduction(X,y)
  }
  
  # reduce data set dimensionality
  rmf_pipeline_name = run_cfg['preproc/rmf/pipeline']
  X = rmf_pipelines[rmf_pipeline_name](X,y)

  flag_normalize = run_cfg['preproc/normalize/enabled']
  if flag_normalize: 
    X = normalize(X)

  # apply pca (with min max normalization)
  if run_cfg['preproc/rmf/pca/enabled']:
    if not flag_normalize: 
      logging.error('Unnormalized data as PCA input!')
    n_comp = run_cfg['preproc/rmf/pca/n_comp']
    X = pca_dim_reduction(X, n_comp)


  ###################### Regression ##############################
  tasks = run_cfg['tasks']
  estimators = cfg_to_estimators(run_cfg)

  X_train, X_test, y_train, y_test = train_test_split(
    X,y ,test_size=run_cfg['overfit/test_size'])

  task_args = [{
    'X': X_train.copy(deep=True),
    'y': y_train.copy(deep=True),
    'task': t
  } for t in tasks]
  args = [{'estimators':estimators,'task_args': a} for a in task_args]

  train_scores = []
  for i, arg in enumerate(args):
    s, m = pool_f(arg)
    train_scores.append(s)

  train_scores_mean = pd.DataFrame( np.array(train_scores).T).mean()
  logging.info(train_scores_mean)

  X_u = X_u[X_train.columns]
  X_u[:] = fill_nan(X_u, run_cfg['preproc/imputer/strategy'])

  for t_name in tasks:
    model_dict = estimators[t_name]
    model = model_dict['model']() # factory
    fit_f = model_dict['fit'](model,X_train,y_train)
    logging.info(model.score(X_test, y_test))
    y_u = model.predict(X_u)
    if len(y_u.shape) > 1:
      yuf = y_u.flatten()
      y_u = yuf
    y_u_df =  pd.DataFrame({
      'id': np.arange(0,len(y_u)).astype(float),
      'y': y_u
    })

    if not os.path.exists('predictions'):
      os.makedirs('predictions')
    y_u_df.to_csv(f'predictions/{t_name}_y.csv', index=False)
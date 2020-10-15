#%%
# should provide auto reload
# %load_ext autoreload 
# %autoreload 2
# Path hack.import sys, os
sys.path.insert(0, os.path.abspath('..'))

#%%
# Other modules
import pandas as pd
import numpy as np
import multiprocessing

import seaborn as sns
import lightgbm as lgbm


from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from sklearn.model_selection import \
  RepeatedKFold, \
  cross_val_score, \
  train_test_split

from sklearn.linear_model import \
  LinearRegression, \
  Lasso, \
  Ridge, \
  ElasticNet

from sklearn.base import BaseEstimator

from sklearn.preprocessing import Normalizer 

from autofeat import FeatureSelector, AutoFeatRegressor

# Our code
from modules import ConfigLoader
import project1_raffi.main as raffi

# fix random gen
random_state = 8


#%% some local testing:
run_cfg = ConfigLoader().from_file('base_cfg.yml')
env_cfg = ConfigLoader().from_file('../env/env.yml')


#%%
print(env_cfg)

#%%
df_X = pd.read_csv(f"{env_cfg['datasets/project1/path']}/X_train.csv")
df_Y = pd.read_csv(f"{env_cfg['datasets/project1/path']}/y_train.csv")
df_X_u = pd.read_csv(f"{env_cfg['datasets/project1/path']}/X_test.csv") # unlabeled 

#%% 
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
normalizer = Normalizer()
df_X[:] = imputer.fit_transform(df_X)

# %%
fsel = FeatureSelector(verbose=1)
df_X_f = fsel.fit_transform(df_X.iloc[:,1:], df_Y['y'])
# df_X_f[:] = normalizer.fit_transform(df_X_f)

# %%
def simple_fit(model, X, y):
  model = model.fit(X, y)
  return model 

def auto_crossval(model, X, y):
  rkf = RepeatedKFold(
    n_splits=10, n_repeats=2,
    random_state=random_state
  )

  scores = cross_val_score(
    model, X, y, cv=rkf, verbose=1,
    scoring='r2'
  )

  return scores

# early stopping
# voting regressor
# adaboostregressor (base: DecisionTreeRegressor)
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

tasks = ['lasso']
tasks = ['lasso', 'ridge', 'lightgbm', 'elasticnet']
# %%
# for train, test in rkf.split(df_X_f):
  # print("%s %s" % (train, test))
def pool_f(args):
  model_dict = estimators[args['task']]
  model = model_dict['model']() # factory
  crossval_fit = model_dict['crossval_fit']
  X = args['X']
  y = args['y']
  return (crossval_fit(model, X, y), model)


#%% train test
X_train, X_test, y_train, y_test = train_test_split(
  df_X_f, df_Y['y'], test_size=0.05)


#%%
args = [{
  'X': X_train.copy(deep=True),
  'y': y_train.copy(deep=True),
  'task': t
} for t in tasks]


#%% compute train scores 
train_scores = []
for i, arg in enumerate(args):
  s, m = pool_f(arg)
  train_scores.append(s)

# test_scores = []
# for i, task in enumerate(tasks):
#   model = trained_models[task]
#   test_scores.append(estimators[task]['validate'](model, X_test, y_test))

#%%
train_scores_mean = pd.DataFrame( np.array(train_scores).T).mean()
results_stat = pd.DataFrame([train_scores_mean])

#%%
# multiprocessing does not work in the interactive
# interpreter (stackoverflow)
# with multiprocessing.Pool(1) as p: 
  # results = p.map(pool_f, args)


def fill_nan(X, strategy):
  imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
  return(imputer.fit_transform(X))

for t_name in tasks:
  model_dict = estimators[t_name]
  model = model_dict['model']() # factory
  fit_f = model_dict['fit'](model,X_train,y_train)
  print(model.score(X_test, y_test))
  X_u = df_X_u[X_train.columns]
  X_u[:] = fill_nan(X_u, 'median')
  y_u = model.predict(X_u)
  y_u_df =  pd.DataFrame({
    'id': np.arange(0,len(y_u)).astype(float),
    'y': y_u
  })
  
  if not os.path.exists('predictions'):
    os.makedirs('predictions')
  y_u_df.to_csv(f'predictions/{t_name}_y.csv', index=False)


# %%
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from project1 import main as m
# %%
class GridSearchEstimator(BaseEstimator):

  def __init__(self, slice, base_cfg, env_cfg):
    self.base_cfg = base_cfg
    self.env_cfg = env_cfg

  def fit(self, X, y):
    m.run(run_cfg, env_cfg)

  def predict(self, X, y):
    check_is_fitted()
  
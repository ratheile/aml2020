#%%
# should provide auto reload
# %load_ext autoreload 
# %autoreload 2
# Path hack.import sys, os
sys.path.insert(0, os.path.abspath('..'))


#%%
from project2_raffi.visualization import pca
from project2_raffi.balancing import balance_dataset
from modules import ConfigLoader

# Other modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% some local testing:
run_cfg = ConfigLoader().from_file('base_cfg.yml')
env_cfg = ConfigLoader().from_file('../env/env.yml')

#%%
print(env_cfg)

#%%
df_X = pd.read_csv(f"{env_cfg['datasets/project2/path']}/X_train.csv")
df_y = pd.read_csv(f"{env_cfg['datasets/project2/path']}/y_train.csv")
df_X_u = pd.read_csv(f"{env_cfg['datasets/project2/path']}/X_test.csv") # unlabeled
# %%

# Remove index column
X = df_X.iloc[:,1:]
y = df_y.iloc[:,1:]
# %%

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

pca = KernelPCA(kernel='linear',  n_components=3)
# tsne = TSNE(n_components=3)
X_pca = pca.fit_transform(X)
# X_tsne = tsne.fit_transform(X)

pc_names = ['pc1', 'pc2', 'pc3']
X_dr = pd.DataFrame(X_pca, columns=pc_names)

X_dr

# %%

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis",fontsize=20)
targets = [0,1,2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = y['y'] == target
    plt.scatter(X_dr.loc[indicesToKeep, 'pc1']
               , X_dr.loc[indicesToKeep, 'pc2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
# %%
import plotly.express as px
Xy_dr = pd.merge(X_dr, y, right_index=True, left_index=True)
fig = px.scatter_3d(Xy_dr,
  x='pc1', y='pc2', z='pc3',
  color='y'
)
fig




# %%
Xy_dr

# %%
from imblearn.over_sampling import \
  SMOTE, \
  SVMSMOTE, \
  SMOTENC, \
  BorderlineSMOTE, \
  KMeansSMOTE, \
  ADASYN, \
  RandomOverSampler


from imblearn.combine import \
  SMOTEENN, \
  SMOTETomek

# import loras
from sklearn.model_selection import \
    RepeatedKFold, \
    cross_val_score, \
    train_test_split

# %%
class LorasSampler():

  def __init__(self):
    pass
  #   min_class_points = features_1_trn
  #   maj_class_points = features_0_trn
  #   k = 30
  #   num_shadow_points = 100
  #   sigma = [.005]*min_class_points.shape[1]
  #   num_generated_points = (len(features_0)-len(features_1))//len(features_1)
  #   num_aff_comb = 300
  #   seed = 42

  # def fit_resample(X, y):
  #   loras_min_class_points = loras.fit_resample(maj_class_points, 
  #                                               min_class_points, k=k, 
  #                                               num_shadow_points=num_shadow_points, 
  #                                               num_generated_points=num_generated_points,
  #                                               num_aff_comb=num_aff_comb)
  #   print(loras_min_class_points.shape)
  #   LoRAS_feat = np.concatenate((loras_min_class_points, maj_class_points))
  #   LoRAS_labels = np.concatenate((np.zeros(len(loras_min_class_points))+1, 
  #                                 np.zeros(len(maj_class_points))))
  #   print(LoRAS_feat.shape)
  #   print(LoRAS_labels.shape)## SMOTE and its extensions oversampling

# %%
from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def get_metrics(y_test, y_pred, y_prob):
    metrics = []
    metrics.append(f1_score(y_test, y_pred, average='micro'))
    metrics.append(balanced_accuracy_score(y_test, y_pred))
    # metrics.append(average_precision_score(y_test, y_prob[:,1]))
    return metrics

def knn(X_train,y_train,X_test,y_test):
    knn = KNeighborsClassifier(n_neighbors=29)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)
    return get_metrics(y_test, y_pred, y_prob)

def lr(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression(
      C=1e5,
      solver='lbfgs',
      multi_class='ovr',
      class_weight={0: 1, 1: 1, 2:1}
    )
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)
    return get_metrics(y_test, y_pred, y_prob)


random_state = 42

# Oversampling
# KMeansSMOTE,. SMOTENC, RandomOverSampler

# Over and undersampling
# SMOTETomek, SMOTENN
oversamplers = {
  'NoUpsampling': None,
  'SMOTE': SMOTE(random_state=random_state, k_neighbors=12), 
  'B1SMOTE' : BorderlineSMOTE(random_state=random_state, k_neighbors=12, kind='borderline-1'),
  'B2SMOTE' : BorderlineSMOTE(random_state=random_state, k_neighbors=12, kind='borderline-2'),
  'SVMSMOTE': SVMSMOTE(random_state=random_state, k_neighbors=12),
  'ADASYN':  ADASYN(random_state=random_state, n_neighbors=12),
  'KMeansSMOTE': KMeansSMOTE(random_state=random_state),
  # 'SMOTENC': SMOTENC(random_state=random_state), # nominal + cont. data
  'RandomOverSampler': RandomOverSampler(random_state=random_state),
  'SMOTEENN':SMOTEENN(random_state=random_state),
  'SMOTETomek': SMOTETomek(random_state=random_state)
}
  # 'LORAS':  LorasSampler()


#%%

met_names = ['f1_score', 'balanced_accuracy_score']
            #  'average_precision_score']

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3
)

#%%
def plot_upsampling(X):
  pca = KernelPCA(kernel='linear',  n_components=3)
  X_pca = pca.fit_transform(X)
  pc_names = ['pc1', 'pc2', 'pc3']
  Xy_dr = pd.merge(X_dr, y, right_index=True, left_index=True)
  fig = px.scatter_3d(Xy_dr,
    x='pc1', y='pc2', z='pc3',
    color='y'
  )
  return fig


#%%
sampler0 = oversamplers['ADASYN']
X_o, y_o = sampler0.fit_resample(
  X_train.copy(deep=True),
  y_train.copy(deep=True)
)


#%%
for key, sampler in oversamplers.items():

  if sampler is not None:
    X_o, y_o = sampler.fit_resample(
      X_train.copy(deep=True),
      y_train.copy(deep=True)
    )
  else:
    X_o, y_o = X_train.copy(deep=True), y_train.copy(deep=True)

  res_knn = knn(X_o, y_o, X_test, y_test)
  res_lr = lr(X_o, y_o, X_test, y_test)

  print(f'Method: {key}')
  for k, v in zip(met_names, res_knn):
    print(f'KNN | {k}:{v}')

  for k, v in zip(met_names, res_lr):
    print(f'LR  | {k}:{v}')

# %%

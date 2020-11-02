import sys
import os

from .visualization import pca
from .balancing import balance_dataset
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

# Other modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

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

random_state = 42
oversamplers = {
  'SMOTE': SMOTE(random_state=random_state, k_neighbors=12), 
  'B1SMOTE' : BorderlineSMOTE(random_state=random_state, k_neighbors=12, kind='borderline-1'),
  'B2SMOTE' : BorderlineSMOTE(random_state=random_state, k_neighbors=12, kind='borderline-2'),
  'SVMSMOTE': SVMSMOTE(random_state=random_state, k_neighbors=12),
  'ADASYN':  ADASYN(random_state=random_state, n_neighbors=12),
  'KMeansSMOTE': KMeansSMOTE(random_state=random_state),
  'RandomOverSampler': RandomOverSampler(random_state=random_state),
  'SMOTEENN':SMOTEENN(random_state=random_state),
  'SMOTETomek': SMOTETomek(random_state=random_state)
}


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


def oversample(method:str, X, y):
  sampler = oversamplers[method]
  if sampler is not None:
    X_o, y_o = sampler.fit_resample(
      X.copy(deep=True),
      y.copy(deep=True)
    )
    return (X_o, y_o)


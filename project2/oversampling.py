import sys
import os
import logging

from .visualization import pca
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

oversamplers = {
  'SMOTE': lambda : SMOTE(random_state=None, k_neighbors=12), 
  'B1SMOTE' : lambda : BorderlineSMOTE(random_state=None, k_neighbors=12, kind='borderline-1'),
  'B2SMOTE' : lambda : BorderlineSMOTE(random_state=None, k_neighbors=12, kind='borderline-2'),
  'SVMSMOTE': lambda : SVMSMOTE(random_state=None, k_neighbors=12),
  'ADASYN': lambda :  ADASYN(random_state=None, n_neighbors=12),
  'KMeansSMOTE': lambda : KMeansSMOTE(random_state=None),
  'RandomOverSampler': lambda : RandomOverSampler(random_state=None),
  'SMOTEENN': lambda : SMOTEENN(random_state=None),
  'SMOTETomek': lambda : SMOTETomek(random_state=None)
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


def oversample(X, y, method:str):
  logging.info(f'Oversampling: Balancing out the dataset with {method}')
  sampler = oversamplers[method]()
  if sampler is not None:
    X_o, y_o = sampler.fit_resample(
      X.copy(deep=True),
      y.copy(deep=True)
    )
    return (X_o, y_o)


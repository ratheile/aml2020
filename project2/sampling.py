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

balancers = {
  # TODO: parametize oversamplers
  'SMOTE': lambda args: SMOTE(**args), 
  'BSMOTE' : lambda args: BorderlineSMOTE(**args),
  'SVMSMOTE': lambda args: SVMSMOTE(**args),
  'ADASYN': lambda args:  ADASYN(**args),
  'KMeansSMOTE': lambda args: KMeansSMOTE(**args),
  'RandomOverSampler': lambda args: RandomOverSampler(**args),
  'SMOTEENN': lambda args: SMOTEENN(**args),
  'SMOTETomek': lambda args: SMOTETomek(**args)
}

# def plot_upsampling(X,y):
#   pca = KernelPCA(kernel='linear',  n_components=3)
#   X_pca = pca.fit_transform(X)
#   pc_names = ['pc1', 'pc2', 'pc3']
#   Xy_dr = pd.merge(X_pca, y, right_index=True, left_index=True)
#   fig = px.scatter_3d(Xy_dr,
#     x='pc1', y='pc2', z='pc3',
#     color='y'
#   )
#   return fig


def balancing(X, y, args, method:str):
  logging.info(f'Oversampling: Balancing out the dataset with {method}')
  balancer = balancers[method](args)
  if balancer is not None:
    X_o, y_o = balancer.fit_resample(
      X.copy(deep=True),
      y.copy(deep=True)
    )
    return (X_o, y_o)


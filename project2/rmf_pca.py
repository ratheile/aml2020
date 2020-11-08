import logging
import pandas as pd

from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


methods = {
  'kernel-pca': lambda args: KernelPCA(**args),
  'pca' : lambda args: PCA(**args)
}

def pca_dim_reduction(X,y,pca_method, pca_args):
  model = pca_dim_reduction_create_model(pca_method, pca_args)
  X_out = pca_dim_reduction_transform(model, X)
  return (X_out, model)


def pca_dim_reduction_create_model(pca_method, pca_args):
  logging.info(f'Creating PCA ({pca_method}) dimensionality reduction with cfg {pca_args}')
  pca = methods[pca_method](pca_args)
  return pca


def pca_dim_reduction_transform(model, X):
  X_pca = model.fit_transform(X)
  X_out = pd.DataFrame(X_pca, columns=[f'pca{i}' for i in range(X_pca.shape[1])])
  logging.info(f'PCA dim reduction: completed. new shape:{X_pca.shape}')
  return X_out

import logging
from umap import UMAP
import pandas as pd


def umap_dim_reduction(X,y,umap_args):
  model = umap_dim_reduction_create_model(umap_args)
  model.fit(X, y=y)
  X_out = umap_dim_reduction_transform(model, X)
  return (X_out, model)


def umap_dim_reduction_create_model(umap_args):
  logging.info(f'Creating umap imensionality reduction with cfg {umap_args}')
  umap = UMAP(**umap_args)
  return umap


def umap_dim_reduction_transform(model, X):
  X_umap = model.transform(X)
  X_out = pd.DataFrame(X_umap, columns=[f'umap{i}' for i in range(X_umap.shape[1])])
  logging.info(f'umap dim reduction: completed. new shape:{X_umap.shape}')
  return X_out
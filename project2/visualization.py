import logging
import os
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA

def pca(X):
  # pca = PCA(n_components=2)
  pca = KernelPCA(kernel='poly',  n_components=2)
  pca.fit(X)
  X_pca = pca.transform(X)
  return X_pca
import pandas as pd
import numpy as np
import logging

from sklearn.ensemble import IsolationForest

def find_isolation_forest_outlier(X,cont_lim):
  clf = IsolationForest(contamination=cont_lim).fit(X)
  y_pred_train = clf.predict(X)
  outliers = np.array(np.where(y_pred_train==-1))
  logging.info(f"Total number of outliers removed: {outliers.size}")
  outliers = outliers.tolist()
  outliers = [ item for elem in outliers for item in elem]
  return outliers
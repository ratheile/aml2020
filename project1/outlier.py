import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest

def remove_isolation_forest_outlier(df_x, df_y,cont_lim):
  #use Isolation Forest to indentify outliers on the selected features
  iso_Y_arr = df_y.to_numpy()
  iso_X_arr = df_x.to_numpy()


  len_Y_outl = len(iso_Y_arr)
    
  iso = IsolationForest(contamination=cont_lim)
  yhat = iso.fit_predict(df_x)
  mask = yhat != -1
  iso_X_arr, iso_Y_arr = iso_X_arr[mask, :], iso_Y_arr[mask]

  len_Y_inl = len_Y_outl-sum(mask)
  print(f"Outliers removal:")
  print(f"Removing {len_Y_inl} outliers out of {len_Y_outl} datapoints.")
  df_y_iso = pd.DataFrame(data=iso_Y_arr, columns=df_y.columns.tolist())
  df_x_iso = pd.DataFrame(data=iso_X_arr, columns=df_x.columns.tolist())
  return(df_x_iso, df_y_iso)


def find_isolation_forest_outlier(X,y,cont_lim):
  clf = IsolationForest(contamination=cont_lim).fit(X)
  y_pred_train = clf.predict(X)
  outliers = np.array(np.where(y_pred_train==-1))
  print(f"Total number of outliers removed: {outliers.size}")
  outliers = outliers.tolist()
  outliers = [ item for elem in outliers for item in elem]
  X_inliers = X.drop(index=outliers)
  y_inliers = y.drop(index=outliers)
  return X_inliers, y_inliers
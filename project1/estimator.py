from main import run
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator

class GridSearchEstimator(BaseEstimator):

  def __init__(self, slice, run_cfg, env_cfg):
    self.env_cfg = env_cfg
    self.run_cfg = run_cfg

  def fit(self, X, y):
    run(self.run_cfg, self.env_cfg)

  def predict(self, X, y):
    check_is_fitted(self)

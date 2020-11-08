#%%
# should provide auto reload
# %load_ext autoreload 
# %autoreload 2
# Path hack.import sys, os
sys.path.insert(0, os.path.abspath('..'))


#%%
from project2_raffi.visualization import pca
from modules import ConfigLoader

# Other modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
from sklearn.preprocessing import StandardScaler
X = df_X.iloc[:,1:]
y = df_y.iloc[:,1:].values.ravel()
X_u = df_X.iloc[:,1:]

scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X_u = scaler.transform(X_u)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.3)

# %%
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklego.mixture import GMMOutlierDetector
from umap import UMAP


from pandas.plotting import parallel_coordinates
from sklego.decomposition import UMAPOutlierDetection, PCAOutlierDetection

def plot_model(mod,X_orig, components, threshold):
    mod = mod(n_components=components, threshold=threshold).fit(X_orig)
    X = X_orig.copy()
    outlier_labels = mod.predict(X)
    X = PCA(n_components=5).fit_transform(X)
    X = pd.DataFrame(X)
    X['label'] = outlier_labels

    
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,3))

    ax = axes[0]
    parallel_coordinates(X.loc[lambda d: d['label'] == 1], class_column='label', alpha=0.5,  ax=ax)
    parallel_coordinates(X.loc[lambda d: d['label'] == -1], class_column='label', color='red', alpha=0.3, ax=ax)
    ax.set_title("outlier shown via parallel coordinates")

    if components == 2:
        ax = axes[1]
        X_reduced = mod.transform(X_orig)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=X['label'])
    return f, ax




# gmm_ol = GMMOutlierDetector(n_components=18, threshold=0.95).fit(X)
# mask = gmm_ol.predict(X)

# %%
# X = X[mask == 1]
# y = y[mask == 1]


# %%
n_comp = 2
umap = UMAP(n_neighbors=16, n_components=n_comp)
X_umap_raw = umap.fit_transform(X)
X_umap = pd.DataFrame(X_umap_raw, 
  columns=[f'c{c}' for c in range(n_comp)])
X_umap['label'] = y

n_km_clusters = 2
kmeans = KMeans(n_clusters=n_km_clusters)
km_label = kmeans.fit_predict(X_umap_raw)


#%% UMAP parallel coordinates plot
f, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,3))
colors=['red', 'geen', 'blue']
for i in [2, 1, 0]:
  ax = axes
  parallel_coordinates(X_umap.loc[lambda d: d['label'] == i],
    class_column='label',
    color=colors[i],
    alpha=0.1,  ax=ax)

X_umap['km_label'] = km_label


# %% UMAP 2d visualization
f, axes = plt.subplots(nrows=1, ncols=1)
targets = [0,1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = km_label == target
    plt.scatter(X_umap.loc[indicesToKeep, 'c0']
               , X_umap.loc[indicesToKeep, 'c1'], c = color, s = 50)

# %% filter out majority cluster 
class_counts = [np.sum(y == i) for i in range(3)]
large_class_index = np.argmax(class_counts)

minority_cluster = np.argmax(
[ X_umap.loc[lambda d:  (d['label'] != large_class_index) & \
                      (d['km_label'] == i)].shape[0] 
for i in range(n_km_clusters) ])

assert X_umap.shape[0] == 4800 - X_test.shape[0]
mask = X_umap['km_label'] == minority_cluster
X_2 = X[mask]
y_2 = y[mask]

# %% visualize minority cluster

from dml import DMLMJ, KDA, LLDA, NCA

def pca_lda(X,y):
  pca = KernelPCA(kernel='linear',  n_components=200)
  X_pca = pca.fit_transform(X_2)
  lda = LinearDiscriminantAnalysis(n_components=2)
  X_lda = lda.fit(X_pca, y_2).transform(X_pca)
  return X_lda

# dmlmj = DMLMJ(num_dims=2)
# X_dmlmj = dmlmj.fit_transform(X_2,y_2)


# kda = KDA(n_components=2, kernel='poly')
# X_kda = kda.fit_transform(X_2, y_2)

algos = {
#  'llda': lambda n: LLDA(n_components=n),
#  'dmlmj':lambda n,X,y: DMLMJ(num_dims=n).fit_transform(X,y),
#  'pca_lda': lambda n,X,y: pca_lda(X,y),
 'lda': lambda n,X,y: LinearDiscriminantAnalysis(n_components=2).fit(X,y).transform(X)
  #  'nca': NCA(num_dims=2)
}

# lda = LinearDiscriminantAnalysis(
#   solver='eigen',
#   shrinkage='auto', 
#   n_components=2)
# X_lda = lda.fit(X_2, y_2).transform(X_2)


# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X_2)

# clf = QuadraticDiscriminantAnalysis()
# X_qda = clf.fit(X, y).transform(X)


for algo_name, algo_f in algos.items():


  # 2d
  X_algo = algo_f(2, X_2, y_2)
  X_dr_2d = pd.DataFrame(X_algo, columns=['pc1', 'pc2'])

  plt.figure()
  plt.figure(figsize=(10,10))
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=14)
  plt.xlabel('Component - 1',fontsize=20)
  plt.ylabel('Component - 2',fontsize=20)
  plt.title(f"{algo_name}",fontsize=20)
  targets = [0,1,2]
  colors = ['r', 'g', 'b']
  for target, color in zip(targets,colors):
      indicesToKeep = y_2 == target
      plt.scatter(X_dr_2d.loc[indicesToKeep, 'pc1']
                , X_dr_2d.loc[indicesToKeep, 'pc2'], c = color, s = 50)

  plt.legend(targets,prop={'size': 15})


  # 3d
  # X_algo = algo_f(2, X_2, y_2)
  # pc_names = ['pc1', 'pc2', 'pc3']
  # X_dr_3d = pd.DataFrame(X_algo, columns=pc_names)
  # import plotly.express as px
  # y_2_df = pd.DataFrame(y_2, columns=['y'])
  # Xy_dr = pd.merge(X_dr_3d, y_2_df, right_index=True, left_index=True)
  # fig = px.scatter_3d(Xy_dr,
  #   x='pc1', y='pc2', z='pc3',
  #   color='y'
  # )
  # fig
  # print(f'Computation of {algo_name} completed')




# %%
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

classifiers={
  'SVC': lambda: SVC(
    C=1.0, 
    kernel='rbf', 
    degree=3, # ignored if kernel not poly
    gamma='scale', 
    coef0=0.0, # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    shrinking=True, # What it this?
    probability=False, 
    tol=0.001, 
    #class_weight='balanced',
    # class_weight={0: weights[0], 1: weights[1]*1.2, 2: weights[2]},
    verbose=True, 
    max_iter=-1, # no limit
    decision_function_shape='ovo', #ovo or ovr
    break_ties=False, 
    random_state=None),
}

lda = LinearDiscriminantAnalysis(n_components=2).fit(X_2, y_2)
X_2_lda = lda.transform(X_2)


# Choose one classifier and train
clf = classifiers['SVC']()
clf = clf.fit(X_2_lda,y_2)


# %%
def predict(kmeans, umap, lda, svc, minority_cluster_label, large_class_index, X):
  majority_cluster_label = 1 - minority_cluster_label
  X_umap = umap.transform(X)
  km_labels = kmeans.predict(X_umap)

  minority_mask = km_labels == minority_cluster_label
  y = (km_labels == majority_cluster_label) * large_class_index

  X_m = X[minority_mask]
  X_m_lda = lda.transform(X_m)
  y_m = svc.predict(X_m_lda)
  y[minority_mask] = y_m
  return y

y_pred = predict(kmeans, umap, lda, clf, minority_cluster, large_class_index, X_test)

# %%
BMAC = balanced_accuracy_score(y_test, y_pred)
print(f'BMAC: {BMAC}')


# %%
y_u = predict(kmeans, umap, lda, clf, minority_cluster, large_class_index, X_u)

# %%
print("Preparing submission ...")
submissions =  pd.DataFrame({
  'id': np.arange(0,len(y_u)).astype(float),
  'y': y_u
})
submissions.to_csv(f'submission_svc.csv', index=False)

# %%

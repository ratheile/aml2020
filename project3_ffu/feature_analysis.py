#%% Imports
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

import os
repopath = '/Users/francescofusaro/Documents/aml20/project3/hash_dir'

from biosppy.signals import ecg
from ecgdetectors import Detectors
from hrv import HRV
import neurokit2 as nk


#%% Load X_train and y_train pickles
def load_data(repopath):
    X_file = "3df9d199aaaef0c4d43a578651f1021fd0638bd2_db9d567589f0c3fc77e5a94bba22793350402cb0ae0827534666545fe12d4884_X.pkl" #S_3
    y_file = "cc1f125a07fa6cb2194c679b212f020229c2ac17_db9d567589f0c3fc77e5a94bba22793350402cb0ae0827534666545fe12d4884_y.pkl"
    
    X = pd.read_pickle(f'{repopath}/{X_file}')
    y = pd.read_pickle(f'{repopath}/{y_file}')
    
    return(X,y)

#%% Check X
def check_X_y(X,y):
    print(X.head(3))
    print(X.columns)
    print(X.index)
    print(X.isnull().values.any())
    print(X.iloc[:,0])
    
    print(y.head(3))
    print(y.index)
    print(y.isnull().values.any())
    
    return
# %% Class specific dataframes
#%%Split Classes
def split_classes(X,y):
    class0_ls = y.index[y['y'] == 0] #healthy
    class1_ls = y.index[y['y'] == 1] #Arrhythmia1
    class2_ls = y.index[y['y'] == 2] #Arrhythmia2
    class3_ls = y.index[y['y'] == 3] #Noise
    
    X0 = X.loc[class0_ls,:] #index is a float (iloc does not work in this case)
    
    X1 = X.loc[class1_ls,:]
    
    X2 = X.loc[class2_ls,:]
    
    X3 = X.loc[class3_ls,:]
    
    return(X0, X1, X2, X3)

#%% plot confusion matrix
def plot_cm(classifier, X_test, y_test):
    np.set_printoptions(precision=2)
    
    class_names = ['c_0','c_1', 'c_2', 'c_3']
    
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
    
        print(title)
        print(disp.confusion_matrix)
    
    plt.show()

#%% reduce X to most most important features
def reduce_X_feat(X):
    
    #drop features from X
    
    #original columns
    #['ECG_Quality_Mean', 'ECG_Quality_STD', 'ECG_Rate_Mean', 'ECG_HRV',
    #   'R_P_biosppy', 'P_P/R_P', 'Q_P/R_P', 'R_P_neurokit', 'S_P/R_P',
    #   'T_P/R_P', 'P_Amp_Mean', 'P_Amp_STD', 'S_Amp_Mean', 'S_Amp_STD',
    #   'QRS_t_Mean', 'QRS_t_STD']
    
    col_name = (['ECG_Quality_Mean', 'ECG_Quality_STD',
       'R_P_biosppy', 'P_P/R_P', 'Q_P/R_P', 'R_P_neurokit', 'S_P/R_P',
       'T_P/R_P'])
    X_red = X.drop(columns=col_name,inplace=False)
    return X_red
#%% make a pair plot
def plot_pp(X_red,y):
    Xy_red = X_red
    Xy_red['y'] = y['y']
    
    sns.pairplot(Xy_red, hue='y')
#%% plot umap embedding
def plot_umap_emb(embedding,y):
    plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in y.y.map({0:0, 1:1, 2:2, 3:3})])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Xred dataset', fontsize=24)

#%% ################## Main #############################

#load data
X, y = load_data(repopath)

#check X, y
check_X_y(X,y)

#split classes
(X0, X1, X2, X3) = split_classes(X, y)

#describe
X0.describe()

#split data into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#run classifier
classifier = svm.SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced').fit(X_train, y_train)

#plot confusion matrix
plot_cm(classifier,X_test,y_test)

#%% reduce X to most important features
X_red = reduce_X_feat(X)
#%%make pair plot of most import features
plot_pp(X_red,y)
#%% umap embedding on X_red
from sklearn.decomposition import PCA
embedding = PCA(n_components=2).fit_transform(X_red)
#%% plot umap embedding
plot_umap_emb(embedding,y)
# %%

# Autoreload 
import autoreload
%load_ext autoreload
%autoreload 2
#%%
import numpy as np
#from keras.utils import to_categorical
#from tqdm import tqdm, tqdm_notebook
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

#from skimage import filters
from sklearn.model_selection import train_test_split

import json
import glob

from keras import optimizers
import keras_metrics


#%% Import modules
from data_generator import CorrDatasetV2
from utils import visualize_plt#, visualize_3d_discr
from model import Model 


#%%
# Main for data generation 
discr_size_fd = 40
scale_code = 40

tau = [0, 2]
dopp = [-2500, 2500]

delta_tau = [0.1, 0.8]
delta_dopp = [250, 1000]
delta_phase = 0
alpha_att = [0.5, 0.9]
Tint = 1e-3

cn0_log=30

#%% Check CorrDataset
Dataset = CorrDatasetV2(discr_size_fd=discr_size_fd,
                        scale_code=scale_code,
                        Tint=Tint,
                        multipath_option=False,
                        delta_tau_interv=delta_tau, 
                        delta_dopp_interv=delta_dopp,
                        delta_phase=delta_phase,
                        alpha_att_interv=alpha_att,
                        tau=tau, dopp=dopp,
                        cn0_log=cn0_log)

# generate 1 peak to check
#matrix, x, y = Dataset.generate_peak()

# generate 10 samples
#samples, module, delta_doppi, delta_taui, alpha_atti = Dataset.build(nb_samples=1)
samples, module = Dataset.build(nb_samples=1)

visualize_plt(samples[0]['table'][...,0])
visualize_plt(samples[0]['table'][...,1])
visualize_plt(module)

#%% Visualize peaks with plotly
#for channel in ['I', 'Q', 'module']:
#    for delta_phase in [0, np.pi/4, np.pi/2, np.pi]:

channel = 'module'
delta_phase = 0
    
Dataset = CorrDatasetV2(discr_size_fd=discr_size_fd,
                    scale_code=scale_code,
                    Tint=Tint,
                    multipath_option=False,
                    delta_tau_interv=delta_tau, 
                    delta_dopp_interv=delta_dopp,
                    delta_phase=delta_phase,
                    alpha_att_interv=alpha_att,
                    tau=tau, dopp=dopp)

samples, module = Dataset.build(nb_samples=1)

filename = 'visu_plotly/plotly_visu_phase-{}_channel-{}.html'.format(delta_phase, channel)

img_dict = {'I': samples[0]['table'][...,0], 
            'Q': samples[0]['table'][...,1],
            'module': module}

visualize_3d_discr(func=img_dict[channel],
                   discr_size_fd=discr_size_fd,
                   scale_code=scale_code,
                   tau_interv=tau, 
                   dopp_interv=dopp,
                   Tint=Tint,
                   delta_dopp=0,
                   delta_tau=0,
                   alpha_att=0,
                   delta_phase=delta_phase,
                   filename=filename)

#%% Check reference feature extractor
Dataset_mp = CorrDatasetV2(discr_size_fd=discr_size_fd,
                        scale_code=scale_code,
                        Tint=Tint,
                        multipath_option=True,
                        delta_tau_interv=delta_tau, 
                        delta_dopp_interv=delta_dopp,
                        delta_phase=delta_phase,
                        alpha_att_interv=alpha_att,
                        tau=tau, dopp=dopp)
Dataset_nomp = CorrDatasetV2(discr_size_fd=discr_size_fd,
                        scale_code=scale_code,
                        Tint=Tint,
                        multipath_option=False,
                        tau=tau, dopp=dopp)

#%%
# generate 1 peak to check
#matrix, x, y = Dataset.generate_peak()

# generate 10 samples
samples, ref_data_samples_mp, module_mp, delta_doppi, delta_taui, alpha_atti = Dataset_mp.build(nb_samples=1, ref_features=True)   
visualize_plt(module_mp[...,0])
print(ref_data_samples_mp)
samples, ref_data_samples_nomp, module_nomp = Dataset_nomp.build(nb_samples=1, ref_features=True)   
visualize_plt(module_nomp[...,0])
print(ref_data_samples_nomp)

# create DF from ref_data_samples dict
df_mp = pd.DataFrame(list(ref_data_samples_mp))
df_nomp = pd.DataFrame(list(ref_data_samples_nomp))
df = pd.concat([df_mp, df_nomp])

#%%
X, y = preprocess_df(df)
df_av = pd.DataFrame(X, columns=['f2', 'f3'])
df_av['label'] = y
# df.groupby('label').std()
#%% Train/ Val SVM on ref features
features = ['f2', 'f3']
target = ['label']

# shuffle dataset
df_av = df_av.sample(frac=1).reset_index(drop=True)

df_train_val, df_test = train_test_split(df_av, test_size=0.2, random_state=42,
                                         shuffle=True, stratify=df_av['label'])


# Train, test split
Xtrain_val, Xtest, ytrain_val, ytest = df_train_val[features], df_test[features], df_train_val[target], df_test[target] 
Xtrain_val, Xtest, ytrain_val, ytest = Xtrain_val.values, Xtest.values, ytrain_val.values, ytest.values


# define k-fold cross val / SVC
kfold = KFold(n_splits=3, shuffle=True)
model = SVC(kernel='rbf', gamma='auto', verbose=True)

# train SVC
scores = []
for train_index, val_index in kfold.split(Xtrain_val):
    # train_test split
    Xtrain, Xval = Xtrain_val[train_index], Xtrain_val[val_index]
    ytrain, yval = ytrain_val[train_index], ytrain_val[val_index]
    
    #scale features
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain_norm = scaler.transform(Xtrain)
    Xval_norm =scaler.transform(Xval)
    
    model.fit(Xtrain_norm, ytrain)
    scores.append(model.score(Xval_norm, yval))

# check preds / classification report
Xtest_norm = scaler.transform(Xtest)
preds = model.predict(Xtest_norm)
confusion_matrix = pd.crosstab(ytest.T[0], preds, rownames=['Real'], colnames=['Predicted'])
print(confusion_matrix)
print(classification_report(ytest.T[0], preds))


precisions, recalls, _, __ = precision_recall_fscore_support(ytest.T[0], preds, average=None)
precision, recall = precisions[1], recalls[1]
accuracy = accuracy_score(ytest.T[0], preds)

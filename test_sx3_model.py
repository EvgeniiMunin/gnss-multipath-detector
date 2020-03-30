import autoreload
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
from keras import layers
from keras import models
from keras import optimizers
import keras_metrics
from sklearn.model_selection import train_test_split

# callbacks
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import glob
import json
import matplotlib.pyplot as plt
import cv2

#%%
from data_generator_sx3 import SX3Dataset
from data_generator import CorrDatasetV2
from model import Model
from utils import save_model, load_model

#%% prepare sx3 data (only module)
dataset_mp = SX3Dataset(label=1, global_path='sx3_data/outputs/mp')
dataset_nomp = SX3Dataset(label=0, global_path='sx3_data/outputs/no_mp')

data_mp = dataset_mp.build(discr_shape=(20,20))
data_nomp = dataset_nomp.build(discr_shape=(20,20))[:500]

dataset = np.concatenate((data_mp, data_nomp), axis=0)
np.random.shuffle(dataset)

data_train, data_val = train_test_split(dataset, test_size=0.2)

# 1 channel image (only module), add newaxis
X_train_sx = np.array([x['table'] for x in data_train])
X_val_sx = np.array([x['table'] for x in data_val])

y_train_sx = np.array([x['label'] for x in data_train])
y_val_sx = np.array([x['label'] for x in data_val])

#%% prepare data generator data (only module)
global_path_mp_i = 'synth_data/mp/*_i_*'
global_path_mp_q = 'synth_data/mp/*_q_*'
global_path_nomp_i = 'synth_data/no_mp/*_i_*'
global_path_nomp_q = 'synth_data/no_mp/*_q_*'
paths_mp_i = sorted(glob.glob(global_path_mp_i))
paths_mp_q = sorted(glob.glob(global_path_mp_q))
paths_nomp_i = sorted(glob.glob(global_path_nomp_i))
paths_nomp_q = sorted(glob.glob(global_path_nomp_q))

synth_data_samples = []
synth_data_labels = []
discr_shape=(20,20)
for path_mp_i, path_mp_q in zip(paths_mp_i, paths_mp_q):
    matr_i = cv2.resize(pd.read_csv(path_mp_i, sep=',', header=None).values, discr_shape)
    matr_q = cv2.resize(pd.read_csv(path_mp_q, sep=',', header=None).values, discr_shape)
    matr_i = (matr_i - matr_i.min()) / (matr_i.max() - matr_i.min())
    matr_q = (matr_q - matr_q.min()) / (matr_q.max() - matr_q.min())
    matr_i = matr_i[...,None]
    matr_q = matr_q[...,None]
    matr = np.concatenate((matr_i, matr_q), axis=2)
    #matr = matr_i**2 + matr_q**2
    synth_data_samples.append(matr)
    synth_data_labels.append(1)
    
for path_nomp_i, path_nomp_q in zip(paths_nomp_i, paths_nomp_q):
    matr_i = cv2.resize(pd.read_csv(path_nomp_i, sep=',', header=None).values, discr_shape)
    matr_q = cv2.resize(pd.read_csv(path_nomp_q, sep=',', header=None).values, discr_shape)
    matr_i = (matr_i - matr_i.min()) / (matr_i.max() - matr_i.min())
    matr_q = (matr_q - matr_q.min()) / (matr_q.max() - matr_q.min())
    matr_i = matr_i[...,None]
    matr_q = matr_q[...,None]
    matr = np.concatenate((matr_i, matr_q), axis=2)
    #matr = matr_i**2 + matr_q**2
    synth_data_samples.append(matr)
    synth_data_labels.append(0)

synth_data_samples = np.array(synth_data_samples)
synth_data_labels = np.array(synth_data_labels)

X_train_synth, X_val_synth, y_train_synth, y_val_synth = train_test_split(synth_data_samples, synth_data_labels, test_size=0.2, shuffle=True)

#%% Define model.
model = Model(shape=(X_train_synth.shape[1], X_train_synth.shape[2], X_train_synth.shape[3]))

batch_size = 8
train_iters = 20
learning_rate = 1e-4

model.model.compile(loss='binary_crossentropy',
					optimizer=optimizers.Adam(lr=learning_rate),
					metrics=['acc',
						   keras_metrics.precision(),
						   keras_metrics.recall()])

#model.model = load_model('saved_models/sc1_data_gen_train.pkl')
#model.model = load_model('saved_models/sc2_fine_tune.pkl')

#%% Train model: pretrain on data gen
# define callbacks
checkpointer = ModelCheckpoint(filepath='saved_models/mp_detector_model.h5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau()
early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    
history = model.model.fit(
    x=X_train_synth,
    y=y_train_synth,
    validation_data=(X_val_synth, y_val_synth),
    epochs=train_iters,
    batch_size=batch_size,
    callbacks=[reduce_lr, early_stopping, checkpointer]
    )
#save_model(model.model, 'saved_models/sc1_data_gen_train_zoom.pkl')

#%% fine tune model on sx3 data
history_sx = model.model.fit(
        x=X_train_sx,
        y=y_train_sx,
        validation_data=(X_val_sx, y_val_sx),
        epochs=train_iters,
        batch_size=batch_size
        )
#save_model(model.model, 'saved_models/sc2_fine_tune_zoom.pkl')

#%% Evaluate model on sx3 validation data

model.model.evaluate(
        x=X_train_sx,
        y=y_train_sx,
        batch_size=batch_size,
        verbose=1
        )

#%% visually compare matrices

#matr_i = pd.read_csv(path_mp_i, sep=',', header=None).values.copy().resize((20,20))
#matr_i = matr_i[...,None]

n = np.random.randint(0, 10)

plt.figure()
print(y_train_synth[n])
plt.imshow(X_train_synth[n,...,0])
plt.figure()
print(y_train_sx[n])
plt.imshow(X_train_sx[n,...,0])

#%% compare histograms between sx3 and data sampler
plt.figure()
plt.hist(X_train_synth.mean(axis=0)[...,0], bins=8)
plt.show()
plt.figure()
plt.hist(X_train_sx.mean(axis=0)[...,0], bins=8)
plt.show()
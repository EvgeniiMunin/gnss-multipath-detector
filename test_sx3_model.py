import autoreload
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
from keras import layers
from keras import models
from keras import optimizers
import keras_metrics
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# callbacks
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import glob
import json
import matplotlib.pyplot as plt
import cv2

from edafa import ClassPredictor
#%%
from data_generator_sx3 import SX3Dataset
from data_generator import CorrDatasetV2, FakeNoiseDataset
from model import Model
from utils import save_model, load_model

#%%
#def gen_ds_dataset(global_path_i, global_path_q, discr_shape=(70,70), multipath_option=False, module_option=False):
#    paths_i = sorted(glob.glob(global_path_i))
#    paths_q = sorted(glob.glob(global_path_q))
#    synth_data_samples = []
#    synth_data_labels = []
#    label = 1 if multipath_option else 0
#    
#    for path_i, path_q in zip(paths_i, paths_q):
#        # load, resize image
#        matr_i = cv2.resize(pd.read_csv(path_i, sep=',', header=None).values, discr_shape)
#        matr_q = cv2.resize(pd.read_csv(path_q, sep=',', header=None).values, discr_shape)
#        
#        # rotate 90 deg counter clockwise
#        #matr_i = cv2.rotate(matr_i, cv2.ROTATE_90_COUNTERCLOCKWISE)
#        #matr_q = cv2.rotate(matr_q, cv2.ROTATE_90_COUNTERCLOCKWISE)
#        
#        if module_option:
#            print('check module option')
#            matr = (matr_i**2 + matr_q**2)[...,None]
#            matr = (matr - matr.min()) / (matr.max() - matr.min())
#        else:
#            # min max scale ?????
#            matr_i = (matr_i - matr_i.min()) / (matr_i.max() - matr_i.min())
#            matr_q = (matr_q - matr_q.min()) / (matr_q.max() - matr_q.min())
#            
#            # add axis
#            matr_i = matr_i[...,None]
#            matr_q = matr_q[...,None]
#            matr = np.concatenate((matr_i, matr_q), axis=2)
#        
#        synth_data_samples.append(matr)
#        synth_data_labels.append(label)
#        
#    synth_data_samples = np.array(synth_data_samples)
#    synth_data_labels = np.array(synth_data_labels)
#
#    return synth_data_samples, synth_data_labels

def load_ds_data(discr, nb_samples=None):
    '''prepare data generator data '''
    global_path_mp_i = 'synth_data/discr_{}/mp/*_i_*'.format(discr)
    global_path_mp_q = 'synth_data/discr_{}/mp/*_q_*'.format(discr)
    global_path_nomp_i = 'synth_data/discr_{}/no_mp/*_i_*'.format(discr)
    global_path_nomp_q = 'synth_data/discr_{}/no_mp/*_q_*'.format(discr)
    if nb_samples is None:
        paths_mp_i = sorted(glob.glob(global_path_mp_i))
        paths_mp_q = sorted(glob.glob(global_path_mp_q))
        paths_nomp_i = sorted(glob.glob(global_path_nomp_i))
        paths_nomp_q = sorted(glob.glob(global_path_nomp_q))
    else:
        paths_mp_i = sorted(glob.glob(global_path_mp_i))[:nb_samples]
        paths_mp_q = sorted(glob.glob(global_path_mp_q))[:nb_samples]
        paths_nomp_i = sorted(glob.glob(global_path_nomp_i))[:nb_samples]
        paths_nomp_q = sorted(glob.glob(global_path_nomp_q))[:nb_samples]

    synth_data_samples = []
    synth_data_labels = []
    for path_mp_i, path_mp_q in zip(paths_mp_i, paths_mp_q):
        matr_i = pd.read_csv(path_mp_i, sep=',', header=None).values
        matr_q = pd.read_csv(path_mp_q, sep=',', header=None).values
        matr_i = matr_i[...,None]
        matr_q = matr_q[...,None]
        matr = np.concatenate((matr_i, matr_q), axis=2)
        #matr = matr_i**2 + matr_q**2
        synth_data_samples.append(matr)
        synth_data_labels.append(1)

    for path_nomp_i, path_nomp_q in zip(paths_nomp_i, paths_nomp_q):
        matr_i = pd.read_csv(path_nomp_i, sep=',', header=None).values
        matr_q = pd.read_csv(path_nomp_q, sep=',', header=None).values
        matr_i = matr_i[...,None]
        matr_q = matr_q[...,None]
        matr = np.concatenate((matr_i, matr_q), axis=2)
        #matr = matr_i**2 + matr_q**2
        synth_data_samples.append(matr)
        synth_data_labels.append(0)

    synth_data_samples = np.array(synth_data_samples)
    synth_data_labels = np.array(synth_data_labels)

    X_train_synth, X_val_synth, y_train_synth, y_val_synth = train_test_split(synth_data_samples, synth_data_labels, test_size=0.2, shuffle=True)
    
    return X_train_synth, X_val_synth, y_train_synth, y_val_synth

#def f1_score(y_ture, y_pred)

#%% prepare sx3 dataset
dataset_nomp = SX3Dataset(label=0, global_path='sx3_data/valid_sx3_dataset/snap_SX3_5_sat_11_89x81')
dataset_mp1 = SX3Dataset(label=1, global_path='sx3_data/valid_sx3_dataset/snap_SX3_MP_5_sat_01_89x81')
dataset_mp2 = SX3Dataset(label=1, global_path='sx3_data/valid_sx3_dataset/snap_SX3_MP_5_sat_11_89x81')
data_nomp = dataset_nomp.build(discr_shape=(80,80), nb_samples=200)
data_mp1 = dataset_mp1.build(discr_shape=(80,80), nb_samples=100)
data_mp2 = dataset_mp2.build(discr_shape=(80,80), nb_samples=100)

dataset = np.concatenate((data_mp1, data_mp2, data_nomp), axis=0)
np.random.shuffle(dataset)

#data_train, data_val = train_test_split(dataset, test_size=0.1)
data_val = dataset

# 1 channel image (only module), add newaxis
#X_train_sx = np.array([x['table'] for x in data_train])
X_val_sx = np.array([x['table'] for x in data_val])

#y_train_sx = np.array([x['label'] for x in data_train])
y_val_sx = np.array([x['label'] for x in data_val])


## create datasets w/wo multipath for histogram
#dataset_nomp_hist = SX3Dataset(label=0, global_path='sx3_data/outputs/no_mp')
#data_nomp_hist = dataset_nomp_hist.build(discr_shape=(70,70), module_option=False)
#X_nomp_hist_sx = np.array([x['table'] for x in data_nomp_hist])
#y_nomp_hist_sx = np.array([x['label'] for x in data_nomp_hist])
#
#dataset_mp_hist = SX3Dataset(label=0, global_path='sx3_data/outputs/mp')
#data_mp_hist = dataset_mp_hist.build(discr_shape=(70,70), module_option=False)
#X_mp_hist_sx = np.array([x['table'] for x in data_mp_hist])
#y_mp_hist_sx = np.array([x['label'] for x in data_mp_hist])

#%% prepare data generator data
discr = 80
X_train_synth, X_val_synth, y_train_synth, y_val_synth = load_ds_data(discr)


#%% Define model.
model = Model(shape=(X_train_synth.shape[1], X_train_synth.shape[2], X_train_synth.shape[3]))

batch_size = 8
train_iters = 10
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
#checkpointer = ModelCheckpoint(filepath='saved_models/mp_detector_model.h5', verbose=1, save_best_only=True)
#reduce_lr = ReduceLROnPlateau()
#early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
#    
#history = model.model.fit(
#    x=X_train_synth,
#    y=y_train_synth,
#    validation_data=(X_val_synth, y_val_synth),
#    epochs=train_iters,
#    batch_size=batch_size,
#    callbacks=[reduce_lr, early_stopping, checkpointer]
#    )
#save_model(model.model, 'saved_models/sc1_data_gen_train_zoom.pkl')

#%% Train model with augmentations
datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.01,
        zoom_range=[0.9,1.25],
        fill_mode='nearest',
        
        #zca_whitening=True,
        channel_shift_range=0.9,
        #brightness_range=[0.5,1.5]
        )

datagen.fit(X_train_synth)

history = model.model.fit_generator(
        datagen.flow(X_train_synth, y_train_synth, batch_size=batch_size),
        validation_data=(X_val_synth, y_val_synth),
        epochs=train_iters,
        )

#%% fine tune model on sx3 data
#history_sx = model.model.fit(
#        x=X_train_sx,
#        y=y_train_sx,
#        validation_data=(X_val_sx, y_val_sx),
#        epochs=train_iters,
#        batch_size=batch_size21
#        )
#save_model(model.model, 'saved_models/sc2_fine_tune_zoom.pkl')

#%% Evaluate model on sx3 validation data

model.model.evaluate(
        x=X_val_sx,
        y=y_val_sx,
        batch_size=batch_size,
        verbose=1
        )

#%% apply test time augmentations
class myPredictor(ClassPredictor):
    '''inherit predictor class and implement main function'''
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
    
    def predict_patches(self, patches):
        return self.model.predict(patches)
    
# instantiate class with configs
conf = '{"args": ["NO", "BRIGHT", "CONTRAST", "GAUSSIAN", "GAMMA", \
                "FLIP_LR", "FLIP_UD"],\
        "mean": "ARITH"}'
predictor = myPredictor(model.model, conf)    

# predict images
y_pred_aug = predictor.predict_images(X_val_sx)
y_pred_aug = [(y[0]>=0.5).astype(np.uint8) for y in y_pred_aug]

print('Acc with TTA: ', np.mean(y_val_sx == y_pred_aug))

#%% Plot augmentations

plt.imshow(next(iter(datagen.flow(X_train_synth)))[0][...,0])
plt.show()

#%% visually compare matrices

#matr_i = pd.read_csv(path_mp_i, sep=',', header=None).values.copy().resize((20,20))
#matr_i = matr_i[...,None]

n = np.random.randint(0, 50)

plt.figure()
print(synth_nomp_labels[n])
#plt.imshow(X_train_synth[n,...,0])
plt.imshow(X_nomp_hist_sx[n,...,0])
plt.figure()
#print(y_mp_hist_sx[n])
#plt.imshow(X_train_sx[n,...,0])
plt.imshow(synth_nomp_samples[n,...,0])

# make KDE plots of images
import matplotlib.tri as tri

discr_shape = (70, 70)
xi = np.linspace(0, discr_shape[0], discr_shape[0])
yi = np.linspace(0, discr_shape[1], discr_shape[1])

fig = plt.figure()
cntr = plt.contour(xi, yi, synth_mp_samples[n,...,0], levels=14, cmap='RdBu_r')
fig.colorbar(cntr)
plt.title('DS multipath. I channel')
plt.show()

fig = plt.figure()
cntr = plt.contour(xi, yi, synth_mp_samples[n,...,1], levels=14, cmap='RdBu_r')
fig.colorbar(cntr)
plt.title('DS multipath. Q channel')
plt.show()

#%% compare histograms between sx3 and data sampler
import seaborn as sns

sns.set()
dist = sns.distplot(synth_nomp_samples[...,0].flatten())
#dist.axes.set_xlim(0.4, 0.8)
plt.title('DS no multipath')
plt.show()
sns.set()
sns.distplot(X_nomp_hist_sx[...,0].flatten())
plt.title('SX3 no multipath')
plt.show()

sns.set()
dist = sns.distplot(synth_mp_samples[...,0].flatten())
#dist.axes.set_xlim(0.4, 0.8)
plt.title('DS multipath')
plt.show()
sns.set()
sns.distplot(X_mp_hist_sx[...,0].flatten())
plt.title('SX3 multipath')
plt.show()
# .mean(axis=0)

#
#plt.figure()
#plt.hist(synth_nomp_samples.mean(axis=0)[...,0], bins=8)
#plt.show()
#plt.figure()
#plt.hist(X_nomp_hist_sx.mean(axis=0)[...,0], bins=8)
#plt.show()

#%% Check fake noise histogram
fake_noise_generator = FakeNoiseDataset()

nb_samples=13
noise_i_path = r'corr_noise_generator/outputs/i_channel/*.csv'
noise_q_path = r'corr_noise_generator/outputs/q_channel/*.csv'
# read i channel
paths = glob.glob(noise_i_path)
noise_i_samples = fake_noise_generator.build(paths[:nb_samples], discr_shape=(70,70))
# read q channel
paths = glob.glob(noise_q_path)
noise_q_samples = fake_noise_generator.build(paths[:nb_samples], discr_shape=(70,70))

# histogram of I channel of noise
sns.set()
sns.distplot(noise_i_samples.flatten())
plt.title('Corr noise I channel')
plt.show()

#%% Visualize sx3 snapshots
#data_path_i = 'sx3_data/snapshots/*_I_*.csv'
#data_path_q = 'sx3_data/snapshots/*_Q_*.csv'
#paths_i = glob.glob(data_path_i)
#paths_q = glob.glob(data_path_q)
#matr = pd.read_csv(paths_i[0], sep=',', header=None).values

# visualize snapshots after preprocessing
dataset_nomp = SX3Dataset(label=0, global_path='sx3_data/valid_sx3_dataset/snap_SX3_5_sat_11_89x81')
data_nomp = dataset_nomp.build(discr_shape=(80,80), nb_samples=5)

#dataset = np.concatenate((data_mp1, data_mp2, data_nomp), axis=0)
#np.random.shuffle(data_nomp)
#data_train, data_val = train_test_split(data_nomp, test_size=0.2)

# 1 channel image (only module), add newaxis
X_train_sx = np.array([x['table'] for x in data_nomp])

np.savetxt('sx3_data/visu_sx3/sx_i_vis.csv', X_train_sx[0,...,0], delimiter=',')
np.savetxt('sx3_data/visu_sx3/sx_q_vis.csv', X_train_sx[0,...,1], delimiter=',')


#%% visualize ds images
discr = 80
X_train_synth, X_val_synth, y_train_synth, y_val_synth = load_ds_data(discr, nb_samples=5)

np.savetxt('sx3_data/visu_sx3/sx_i_vis_synth.csv', X_train_synth[0,...,0], delimiter=',')
np.savetxt('sx3_data/visu_sx3/sx_q_vis_synth.csv', X_train_synth[0,...,1], delimiter=',')





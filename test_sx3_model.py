import autoreload
%load_ext autoreload
%autoreload 2

import numpy as np
from keras import layers
from keras import models
from keras import optimizers
import keras_metrics
from sklearn.model_selection import train_test_split

import glob
import json

#%%
from data_generator_sx3 import SX3Dataset
from data_generator import CorrDatasetV2
from model import Model
from utils import save_model, load_model

#%% prepare sx3 data
dataset_mp = SX3Dataset(label=1, global_path='sx3_data/sx3_dataset/Courbes_multipath/*')
dataset_nomp = SX3Dataset(label=0, global_path='sx3_data/sx3_dataset/Courbes_all/*')

data_mp = dataset_mp.build()
data_nomp = dataset_nomp.build()[:500]

dataset = np.concatenate((data_mp, data_nomp), axis=0)
np.random.shuffle(dataset)

data_train, data_val = train_test_split(dataset, test_size=0.2)

# 1 channel image, add newaxis
X_train_sx = np.array([x['table'] for x in data_train])[...,None]
X_val_sx = np.array([x['table'] for x in data_val])[...,None]

y_train_sx = np.array([x['label'] for x in data_train])
y_val_sx = np.array([x['label'] for x in data_val])

#%% prepare data generator data
discr_size_fd = 40
scale_code = 40
Tint = 2e-3

configs = []
#allFiles = glob.glob("config_ti{}/config_*.json".format(t_path)) #config_dopp_ph0.json 
#for file_ in allFiles:
file_ = "config_ti1_combin/config_combin_ph0.json"
with open(file_) as json_config_file:
    configs.append(json.load(json_config_file))
print(configs[0])
config = configs[0]

tau = [0, 2]
dopp = [-2000, 2000]

delta_tau = [config['delta_tau_min'], config['delta_tau_max']]
delta_dopp = [config['delta_dopp_min'], config['delta_dopp_max']]
alpha_att = [config['alpha_att_min'], config['alpha_att_max']]
delta_phase = config['delta_phase'] * np.pi / 180
cn0_logs = config['cn0_log']
cn0_log = cn0_logs[3]

dataset = np.array([])
for multipath_option in [True, False]:
    # Build dataset for default branch
    if multipath_option:
        Dataset = CorrDatasetV2(discr_size_fd=discr_size_fd,
									    scale_code=scale_code,
									    Tint=Tint,
									    multipath_option=multipath_option,
									    delta_tau_interv=delta_tau, 
									    delta_dopp_interv=delta_dopp,
									    delta_phase=delta_phase,
									    alpha_att_interv=alpha_att,
									    tau=tau, dopp=dopp,
									    cn0_log=cn0_log)
    else:
        Dataset = CorrDatasetV2(discr_size_fd=discr_size_fd,
									    scale_code=scale_code,
									    Tint=Tint,
									    multipath_option=multipath_option,
									    delta_tau_interv=delta_tau, 
									    delta_dopp_interv=delta_dopp,
									    delta_phase=0,
									    alpha_att_interv=alpha_att,
									    tau=tau, dopp=dopp,
									    cn0_log=cn0_log)
    dataset_temp = Dataset.build(nb_samples=1000)
    # Concatenate and shuffle arrays
    dataset = np.concatenate((dataset, dataset_temp), axis=0)
    
np.random.shuffle(dataset)
data_train, data_val = train_test_split(dataset, test_size=0.2)

X_train = np.array([x['module'] for x in data_train])
X_val = np.array([x['module'] for x in data_val])

y_train = np.array([x['label'] for x in data_train])
y_val = np.array([x['label'] for x in data_val])

#%% Define model.
model = Model(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

batch_size = 16
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
history = model.model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    epochs=train_iters,
    batch_size=batch_size
    )
save_model(model.model, 'saved_models/sc1_data_gen_train_zoom.pkl')

#%% fine tune model on sx3 data
history_sx = model.model.fit(
        x=X_train_sx,
        y=y_train_sx,
        #validation_data=(X_val_sx, y_val_sx),
        epochs=train_iters,
        batch_size=batch_size
        )
save_model(model.model, 'saved_models/sc2_fine_tune_zoom.pkl')

#%% Evaluate model on sx3 validation data

model.model.evaluate(
        x=X_val_sx,
        y=y_val_sx,
        batch_size=batch_size,
        verbose=1
        )

#%% visually compare matrices
import matplotlib.pyplot as plt

n = np.random.randint(0, 100)

plt.figure()
print(y_train[n])
plt.imshow(X_train[n,...,0])
plt.figure()
print(y_train_sx[n])
plt.imshow(X_train_sx[n,...,0])


# Autoreload 
import autoreload
%load_ext autoreload
%autoreload 2
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
import glob
import datetime
import random
#from keras.utils import to_categorical
#from tqdm import tqdm, tqdm_notebook

#%% Import modules
from data_generator import CorrDatasetV2, FakeNoiseDataset
from data_sampler import DataSampler
from utils import visualize_plt#, visualize_3d_discr
#from model import Model 

#%%

def signaltonoise(a, axis=0, ddof=0):
    '''compute SNR for backward compatibility'''
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
    
#%%
# coherent integration period
Tint = 20e-3

# Main for data generation 
discr_size_fd = 40
scale_code = 40

# multipath intervals
delta_tau_interv = [0, 3/2]
delta_dopp_max = min(3/Tint, 800)
delta_dopp_interv = [-delta_dopp_max, delta_dopp_max]
delta_phase = 0
alpha_att_interv = [0.5, 0.9]

# define intervals
# chip rate/ period of PRN code
Fc = 1.023e6
Tc = 1/Fc
Nc = 1023
Fs = 20e6

# doppler interval
dopp_max = min(5.5/Tint, 800+2.5/Tint)
dopp_interval = [-dopp_max, dopp_max]
# length of local PRN code
lC = 20000
# code intervals
tau_interval = [-3/2, 5/2]
tau_prime_interval = [0, 4]


#%% Check CorrDataset
#random.seed(42)
multipath_option = True

Dataset = CorrDatasetV2(discr_size_fd=discr_size_fd,
                        scale_code=scale_code,
                        Tint=Tint,
                        multipath_option=multipath_option,
                        delta_tau_interv=delta_tau_interv,
                        delta_dopp_interv=delta_dopp_interv,
                        delta_phase=delta_phase,
                        alpha_att_interv=alpha_att_interv,
                        tau=tau_prime_interval, 
                        dopp=dopp_interval,
                        cn0_log=cn0_log)

samples = Dataset.build(nb_samples=1)
visualize_plt(samples[0]['table'][...,0])
visualize_plt(samples[0]['table'][...,1])
#visualize_plt(samples[0]['module'][...,0])

#np.savetxt(r'synth_data/valid_data/channel_i_mp-{}.csv'.format(multipath_option), samples[0]['table'][...,0], delimiter=',')
#np.savetxt(r'synth_data/valid_data/channel_q_mp-{}.csv'.format(multipath_option), samples[0]['table'][...,1], delimiter=',')

#%% check adjustment of noise coeff in data sampler

# visualize nomp CN0 50
ipaths = glob.glob('synth_data/no_mp/channel_i*.csv')
qpaths = glob.glob('synth_data/no_mp/channel_q*.csv')

for ind, (ipath, qpath) in enumerate(zip(ipaths, qpaths)):
  synth_data_i = np.genfromtxt(ipath, delimiter=',')
  synth_data_q = np.genfromtxt(qpath, delimiter=',')
  
plt.imshow(synth_data_i)

# check signal to noise ratio
signaltonoise(synth_data_i, axis=None)

#%% Visualize sx3 snapshots
ipaths = glob.glob('sx3_data/visu_sx3/*_i_vis.csv')
qpaths = glob.glob('sx3_data/visu_sx3/*_q_vis.csv')

for ind, (ipath, qpath) in enumerate(zip(ipaths, qpaths)):
    matr = pd.read_csv(ipath, sep=',', header=None).values
    matr = (matr - matr.min()) / (matr.max() - matr.min())
plt.imshow(matr)

# check signal to noise ratio
signaltonoise(matr, axis=None)

#%% Check CorrDataset generation for multipath case
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

# Check MP/ noMP
Dataset_mp = CorrDatasetV2(discr_size_fd=discr_size_fd,
                        scale_code=scale_code,
                        Tint=Tint,
                        multipath_option=True,
                        delta_tau_interv=delta_tau, 
                        delta_dopp_interv=delta_dopp,
                        delta_phase=delta_phase,
                        alpha_att_interv=alpha_att,
                        tau=tau, dopp=dopp)
#Dataset_nomp = CorrDatasetV2(discr_size_fd=discr_size_fd,
#                        scale_code=scale_code,
#                        Tint=Tint,
#                        multipath_option=False,
#                        tau=tau, dopp=dopp)

# generate 1 peak to check
#matrix, x, y = Dataset.generate_peak()

# generate 10 samples
#samples, module, delta_doppi, delta_taui, alpha_atti = Dataset.build(nb_samples=1)
samples = Dataset.build(nb_samples=13)

#visualize_plt(samples[0]['table'][...,0])
#visualize_plt(samples[0]['table'][...,1])
#visualize_plt(module)

#%% check fake noise factor in data sampler 
noise_i_path = r'corr_noise_generator/outputs/i_channel/*.csv'
noise_q_path = r'corr_noise_generator/outputs/q_channel/*.csv'

for cn0_log in [20, 30, 40, 50, 60]:
    print('----', cn0_log)
    data_sampler = DataSampler(
                    discr_size_fd=discr_size_fd,
                    scale_code=scale_code,
                    Tint=Tint,
                    multipath_option=False,
                    delta_tau_interv=delta_tau_interv, 
                    delta_dopp_interv=delta_dopp_interv,
                    delta_phase=delta_phase,
                    alpha_att_interv=alpha_att_interv,
                    tau=tau_interval, 
                    dopp=dopp_interval,
                    cn0_log=cn0_log
                )
    
    nb_samples = 1
    data_sampler.read_noise(noise_i_path, noise_q_path, matrix_shape=(discr_size_fd, scale_code), nb_samples=nb_samples)
    data_sampler.generate_corr(nb_samples=nb_samples)
    matr_i, matr_q = data_sampler.sum_matr(save_csv=False)

plt.imshow(matr_i[0,...])


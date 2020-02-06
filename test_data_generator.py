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
delta_dopp = [-1000, 1000]
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
samples = Dataset.build(nb_samples=1)

#visualize_plt(samples[0]['table'][...,0])
#visualize_plt(samples[0]['table'][...,1])
#visualize_plt(module)

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

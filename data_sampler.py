import numpy as np
import cv2

#import json
import glob
#from keras.utils import to_categorical
#from tqdm import tqdm, tqdm_notebook

# Import modules
import datetime
from data_generator import CorrDatasetV2, FakeNoiseDataset

import matplotlib.pyplot as plt

NOISE_COEF = 1

class DataSampler:
    
    def __init__(self, discr_size_fd, scale_code,    
               Tint=10**-3, 
               multipath_option=False,
               delta_tau_interv=None,
               delta_dopp_interv=None,
               delta_phase=0,
               alpha_att_interv=None,
               tau=[0,2], dopp=[-2000,2000], cn0_log=50):
        
        self.discr_size_fd = discr_size_fd
        self.scale_code = scale_code
        
        self.tau = tau
        self.dopp = dopp
        self.delta_tau_interv = delta_tau_interv
        self.delta_dopp_interv = delta_dopp_interv
        self.alpha_att_interv = alpha_att_interv
        self.delta_phase = delta_phase
        
        self.Tint = Tint
        self.cn0_log = cn0_log
        self.multipath_option = multipath_option
    
        # compute cn0 form cn0_log
        self.cn0 = 10**(0.1*self.cn0_log)
        
    def read_noise(self, i_path, q_path, matrix_shape, nb_samples=13):
        fake_noise_generator = FakeNoiseDataset()
        
        # read i channel
        paths = glob.glob(i_path)
        self.noise_i_samples = fake_noise_generator.build(paths[:nb_samples], discr_shape=matrix_shape)
        # read q channel
        paths = glob.glob(q_path)
        self.noise_q_samples = fake_noise_generator.build(paths[:nb_samples], discr_shape=matrix_shape)
        
        # compute noise factor
        p = (self.noise_i_samples[0]**2 + self.noise_q_samples[0]**2).max()
        var_i = np.var(self.noise_i_samples[0])
        var_q = np.var(self.noise_q_samples[0])
        noise_factor_i = np.sqrt(p / (2 * var_i * self.Tint * self.cn0)) * NOISE_COEF
        noise_factor_q = np.sqrt(p / (2 * var_q * self.Tint * self.cn0)) * NOISE_COEF
        
        #print('check matrix min/max before factor: ', self.noise_i_samples.min(), self.noise_i_samples.max())
        #print('check noise factor: ', noise_factor_i, noise_factor_q)
        #print('check terms: ', p, var_i, self.Tint, self.cn0)
        
        # apply noise factor
        # paths = glob.glob('corr_noise_generator/outputs/i_channel/*.csv')
        # paths = glob.glob('corr_noise_generator/outputs/q_channel/*.csv')
        self.noise_i_samples *= noise_factor_i
        self.noise_q_samples *= noise_factor_q 
        
        self.noise_i_samples = np.transpose(self.noise_i_samples, [0,2,1])
        self.noise_q_samples = np.transpose(self.noise_q_samples, [0,2,1])
        
        #print('check matrix min/max after factor: ', self.noise_i_samples.min(), self.noise_i_samples.max())
    
    def generate_corr(self, nb_samples=13):
        # no_mp/ mp option
        Dataset = CorrDatasetV2(discr_size_fd=self.discr_size_fd,
                            scale_code=self.scale_code,
                            Tint=self.Tint,
                            multipath_option=self.multipath_option,
                            delta_tau_interv=self.delta_tau_interv, 
                            delta_dopp_interv=self.delta_dopp_interv,
                            delta_phase=self.delta_phase,
                            alpha_att_interv=self.alpha_att_interv,
                            tau=self.tau, dopp=self.dopp,
                            cn0_log=self.cn0_log)
        
        samples = Dataset.build(nb_samples=nb_samples)
        # extract separately I,Q channels
        sample_i_func = lambda x: x['table'][...,0]
        sample_q_func = lambda x: x['table'][...,1]
        self.i_samples = np.array(list(map(sample_i_func, samples)))
        self.q_samples = np.array(list(map(sample_q_func, samples)))
    
    def sum_matr(self, save_csv=True):
        #noise_i_samples, noise_q_samples = noise_tuple
        #i_samples, q_samples = sign_tuple
        
        # check matrices shapes
        
        try:
            # check correspondance among arrays shapes
            if (self.i_samples.shape[1] != self.noise_i_samples.shape[1]) or (self.i_samples.shape[2] != self.noise_i_samples.shape[2]):
                print('Wrong arrays shapes. ValueError exception')
                raise ValueError
            
            if self.i_samples.shape[0] != self.noise_i_samples.shape[0]:
                #print('Nb samples correction: ', self.i_samples.shape, self.noise_i_samples.shape)
                min_nb_samples = min(self.i_samples.shape[0], self.noise_i_samples.shape[0])
                self.i_samples = self.i_samples[:min_nb_samples,...]
                self.q_samples = self.q_samples[:min_nb_samples,...]
                self.noise_i_samples = self.noise_i_samples[:min_nb_samples,...]
                self.noise_q_samples = self.noise_q_samples[:min_nb_samples,...]
            
            
            print('CHECK NOISE MODULE MATRIX BEFORE SUM')
            
            matr_i = np.sum([self.i_samples, self.noise_i_samples], axis=0)
            matr_q = np.sum([self.q_samples, self.noise_q_samples], axis=0)
            
            #plt.figure()
            #plt.imshow(matr_i[0,...]**2 + matr_q[0,...]**2)
            #plt.show()
            
            #print('check matrix min/max: ', self.i_samples.min(), self.q_samples.max(), self.noise_i_samples.min(), self.noise_q_samples.max())
            
            # save matrix into csv
            if save_csv:
                #print('check multipath option: ', self.multipath_option)
                #print('check matr_i shape: ', matr_i.shape)
                for i in range(matr_i.shape[0]):
                    print('---------- EXAMPLE {} --------'.format(i))
                    datetime_now = datetime.datetime.now()
                    # save i/q_channel
                    if self.multipath_option:
                        pathi = r'synth_data/mp/channel_i_{}.csv'.format(str(datetime_now))
                        pathq = r'synth_data/mp/channel_q_{}.csv'.format(str(datetime_now))    
                    else:
                        pathi = r'synth_data/no_mp/channel_i_{}.csv'.format(str(datetime_now))
                        pathq = r'synth_data/no_mp/channel_q_{}.csv'.format(str(datetime_now))
                    np.savetxt(pathi, matr_i[i,...], delimiter=',')
                    np.savetxt(pathq, matr_q[i,...], delimiter=',')
            else:
                return matr_i, matr_q
                
        except ValueError:
            print('Wrong arrays shapes: sampels: {}, {}; noise samples {}, {}'.format(self.i_samples.shape, self.q_samples.shape, 
                                                                                                         self.noise_i_samples.shape, self.noise_q_samples.shape))
            
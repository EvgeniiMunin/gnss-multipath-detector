import numpy as np

#import json
import glob
import datetime
#from keras.utils import to_categorical
#from tqdm import tqdm, tqdm_notebook

# Import modules
from data_generator import CorrDatasetV2, FakeNoiseDataset

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
    
    def read_noise(self, i_path, q_path, matrix_shape, nb_samples=13, noise_factor=0.5):
        fake_noise_generator = FakeNoiseDataset(discr=matrix_shape)
        # read i channel 
        # paths = glob.glob('corr_noise_generator/outputs/i_channel/*.csv')
        paths = glob.glob(i_path)
        self.noise_i_samples = fake_noise_generator.build(paths[:nb_samples])
        self.noise_i_samples *= noise_factor

        # read q channel
        # paths = glob.glob('corr_noise_generator/outputs/q_channel/*.csv')
        paths = glob.glob(q_path)
        self.noise_q_samples = fake_noise_generator.build(paths[:nb_samples])
        self.noise_q_samples *= noise_factor   
        #return noise_i_samples, noise_q_samples
    
    def generate_corr(self, nb_samples=13, multipath_option=False):
        # no_mp/ mp option
        Dataset = CorrDatasetV2(discr_size_fd=self.discr_size_fd,
                            scale_code=self.scale_code,
                            Tint=self.Tint,
                            multipath_option=multipath_option,
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
        #return i_samples, q_samples
    
    def sum_matr(self):
        #noise_i_samples, noise_q_samples = noise_tuple
        #i_samples, q_samples = sign_tuple
        
        # check matrices shapes
        try:
            # check correspondance among arrays shapes
            if len(set([self.i_samples.shape, self.q_samples.shape, self.noise_i_samples.shape, self.noise_q_samples.shape])) > 1:
                print('Wrong arrays shapes. ValueError exception')
                raise ValueError
            
            matr_i = np.sum([self.i_samples, self.noise_i_samples], axis=0)
            matr_q = np.sum([self.q_samples, self.noise_q_samples], axis=0)
            
            # save matrix into csv
            for i in range(matr_i.shape[0]):
                # save i_channel
                path = r'synth_data/no_mp/channel_i_{}.csv'.format(str(datetime.datetime.now()))
                np.savetxt(path, matr_i[i,...], delimiter=',')
                # save q_channel
                path = r'synth_data/no_mp/channel_q_{}.csv'.format(str(datetime.datetime.now()))
                np.savetxt(path, matr_q[i,...], delimiter=',')
                
        except ValueError:
            print('Wrong arrays shapes: sampels: {}, {}; noise samples {}, {}'.format(self.i_samples.shape, self.q_samples.shape, 
                                                                                                         self.noise_i_samples.shape, self.noise_q_samples.shape))
            
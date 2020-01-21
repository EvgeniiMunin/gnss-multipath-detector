# Autoreload 
#import autoreload
#%load_ext autoreload
#%autoreload 2

import numpy as np
import math
from scipy import signal
import cv2
#from sklearn.preprocessing import MinMaxScaler

from reference_feature_extractor import FeatureExtractor

class CorrDatasetV2():
  
    def __init__(self, discr_size_fd, scale_code,    
               Tint=10**-3, 
               multipath_option=False,
               delta_tau_interv=None,
               delta_dopp_interv=None,
               delta_phase=0,
               alpha_att_interv=None,
               tau=[0,2], dopp=[-1000,1000], cn0_log=50):
    
        self.discr_size_fd = discr_size_fd
        self.scale_code = scale_code
        
        self.cn0_log = cn0_log # C/N0
        self.Tint = Tint
#        self.b_rf = 10**6 # RF frontend bandwidth
        
        self.multipath_option = multipath_option
        self.delta_tau_interv = delta_tau_interv
        self.delta_dopp_interv = delta_dopp_interv
        self.delta_phase = delta_phase
        self.alpha_att_interv = alpha_att_interv        
        self.tau = tau
        self.dopp = dopp
        
        # claculate SNR
        self.sign_amp = 1
        self.sign_power = 8 * self.sign_amp / self.Tint**2
        self.noise_psd = self.sign_power / 10**(0.1*self.cn0_log)
        
    def sin_cos_matrix(self, multipath=False, delta_dopp=0, delta_phase=0, xk=0, yk=0):
        dopp_axis = np.linspace(start=self.dopp[0],
                                stop=self.dopp[1],
                                num=self.discr_size_fd)
        cos_array = np.array([math.cos(math.pi * (x + delta_dopp) * self.Tint + delta_phase) for x in dopp_axis])
        sin_array = np.array([math.sin(math.pi * (x + delta_dopp) * self.Tint + delta_phase) for x in dopp_axis])
        cos_matrix = np.tile(cos_array, (self.scale_code, 1))
        sin_matrix = np.tile(sin_array, (self.scale_code, 1))
        
        # reshape matrices in case of multipath
        if multipath:
            cos_matrix = cos_matrix[:cos_matrix.shape[0]-xk, :cos_matrix.shape[1]-yk]
            sin_matrix = sin_matrix[:sin_matrix.shape[0]-xk, :sin_matrix.shape[1]-yk]
        return cos_matrix, sin_matrix
            
    def noise_model(self):         
        noise_corr_mean = 0
        noise_corr_std = math.sqrt(self.noise_psd * self.Tint / 16)
        
        #print('snr_log: {}, sign power {}, noise psd: {},  noise corr std: {}'.format(self.snr_log, self.sign_power, noise_psd, noise_corr_std))
        return noise_corr_mean, noise_corr_std
   
    
    def generate_peak(self, multipath=False, delta_dopp=0, delta_tau=0, delta_phase=0, alpha_att=1, ref_features=False):
        x = np.linspace(self.dopp[0], self.dopp[1], self.discr_size_fd)
        y = np.linspace(self.tau[0], self.tau[1], self.scale_code)
        
        # Create empty matrix for peaks
        matrix = np.zeros((self.discr_size_fd, self.scale_code))
        
        # Convert tau/ doppler deviation into pixel scale
        xk = int(x.mean() + delta_dopp / (x.max() - x.min()) * self.discr_size_fd)
        yk = int(y.mean() + delta_tau / (y.max() - y.min()) * self.scale_code) 
        
        print('check deviations: ', xk, yk)
        
        
        # Generate triangle/ sinc function
        func1 = self.sign_amp * signal.triang(self.scale_code)
        func2 = self.sign_amp * np.sinc((x + delta_dopp) * self.Tint)
        
        # Only 1 principal peak
        for i, point in enumerate(func2):
            matrix[i] = alpha_att * func1 * point
        
        # Superpose 2 peaks. Weight matrix of MP peak by the matrix of principal peak 
        if multipath:
            if xk >= 0:
                matrix = matrix[:matrix.shape[0]- xk, :matrix.shape[1]-yk]
            else:
                matrix = matrix[abs(xk):, :matrix.shape[1]-yk]
            print('check mp matrix shape: ', matrix.shape)
        
        # Split matrices in I, Q channels
        print('check cos shape: ', self.sin_cos_matrix(multipath=multipath, delta_dopp=delta_dopp, delta_phase=delta_phase, xk=xk, yk=yk)[0].shape)
        I = matrix * self.sin_cos_matrix(multipath=multipath, delta_dopp=delta_dopp, delta_phase=delta_phase, xk=xk, yk=yk)[0]
        Q = -matrix * self.sin_cos_matrix(multipath=multipath, delta_dopp=delta_dopp, delta_phase=delta_phase, xk=xk, yk=yk)[1]
         
        # Add noise model
        mean = self.noise_model()[0]
        var = self.noise_model()[1]
        

        module = np.sqrt(I**2 + Q**2)
        I += np.random.normal(mean, var, size=matrix.shape)
        Q += np.random.normal(mean, var, size=matrix.shape)
       
        #if ref_features:
        #    print('check no normalization for reference')
        #    I_norm = I[...,None]
        #    Q_norm = Q[...,None]
        #else:
        I_norm = (I - module.min()) / (module.max() - module.min())
        Q_norm = (Q - module.min()) / (module.max() - module.min())
        module = (module - module.min()) / (module.max() - module.min())
        #I_norm = I
        #Q_norm = Q
        #print('CHECK SCALE')
        
        I_norm = I_norm[...,None]
        Q_norm = Q_norm[...,None]

        matrix = np.concatenate((I_norm, Q_norm), axis=2)
       
        return matrix, module, xk, yk
# -----------------------------------------------------------------------------
  
    def build(self, nb_samples=10, ref_features=False, sec_der=False, four_ch=False):
        data_samples = []
#        ref_data_samples = []
        for i in range(nb_samples):
            data = {}
#            ref_data = {}
              
            # Generate matrices: main, multipath
            if self.multipath_option:
                # Set random delta_tau / delta_dopp
                delta_taui = np.random.uniform(low=self.delta_tau_interv[0], high=self.delta_tau_interv[1])
                delta_doppi = np.random.uniform(low=self.delta_dopp_interv[0], high=self.delta_dopp_interv[1])
                alpha_atti = np.random.uniform(low=self.alpha_att_interv[0], high=self.alpha_att_interv[1])
                
                matrix, module, x, y = self.generate_peak()
                matrix_mp, module_mp, x, y = self.generate_peak(multipath=self.multipath_option,
                                                         delta_dopp=delta_doppi, 
                                                         delta_tau=delta_taui,
                                                         delta_phase=self.delta_phase,
                                                         alpha_att=alpha_atti,
                                                         ref_features=ref_features)
                print('check x y: ', x, y)
                print('check matrix shape: ', matrix.shape, matrix[x:, y:].shape, matrix_mp.shape)
                matrix[x:, y:] = matrix[x:, y:] + matrix_mp
                module[x:, y:] = module[x:, y:] + module_mp
            else:
                matrix, module, x, y = self.generate_peak(delta_phase=self.delta_phase, 
                                                          ref_features=ref_features)
                
            data['table'] = matrix
            data['module'] = module[...,None]
            
            # Take into account 2nd derivative computation
            if sec_der:
                data['table_sec_der'] = filter_2der(matrix, kernel_size=5)
                if four_ch:
                    data['table_four_ch'] = np.concatenate((data['table'][...,None], data['table_sec_der'][...,None]), axis=2)
                
            # Generate label
            if self.multipath_option:
                data['label'] = 1
            else:
                data['label'] = 0
              
            data_samples.append(data)
        
        return data_samples
            
            # Compute reference features for given matrix
#            if ref_features:
#                feature_extractor = FeatureExtractor(module.squeeze())
#                ref_data['f2'] = feature_extractor.extract_f2()
#                ref_data['f3'] = feature_extractor.extract_f3(self.tau)
#                # Generate label
#                if self.multipath_option:
#                    ref_data['label'] = 1
#                else:
#                    ref_data['label'] = 0
#                ref_data_samples.append(ref_data)
        
#        if self.multipath_option:
##            if ref_features:
##                self.data_samples = np.array(data_samples)
##                self.ref_data_samples = np.array(ref_data_samples)
##                return self.data_samples, self.ref_data_samples, module, delta_doppi, delta_taui, alpha_atti
##            else:
#            self.data_samples = np.array(data_samples)
#            return self.data_samples, delta_doppi, delta_taui, alpha_atti
#        else:
##            if ref_features:
##                self.data_samples = np.array(data_samples)
##                self.ref_data_samples = np.array(ref_data_samples)
##                return self.data_samples, self.ref_data_samples, module
##            else:
#            self.data_samples = np.array(data_samples)
#            return self.data_samples

def filter_2der(img, kernel_size):
    filt = np.array([1, -2, 1])[:, None]
    img= cv2.medianBlur(np.float32(img), kernel_size)
    img = cv2.filter2D(img, cv2.CV_32F, filt)
    return img

# Autoreload 
#import autoreload
#%load_ext autoreload
#%autoreload 2

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2



class SX3Dataset():
  def __init__(self, label=0, global_path='test_subset/mp_'):
    self.global_path_i = global_path + '/*I*.csv'
    self.global_path_q = global_path + '/*Q*.csv'
    self.label = label
    self.data_samples = []

  def __build_matr__(self, path):
    matr = pd.read_csv(path, sep=',', header=None).values
    #print('check origin matrix shape: ', matr.shape)
    
    # crop and resize matr
    max_ind = np.unravel_index(np.argmax(matr), matr.shape)

    # check max_in position wrt borders (make square matrix)
    if max_ind[0] - matr.shape[1]//2 <= 0:
      matr_crop = matr[:matr.shape[1], :]
    elif max_ind[0] + matr.shape[1]//2 >= 0:
      matr_crop = matr[-matr.shape[1]:, :]
    else:
      matr_crop = matr[max_ind[0]-matr.shape[1]//2 : max_ind[0]+matr.shape[1]//2, :]

    matr_resize = cv2.resize(matr_crop, self.discr_shape)
    
    # scale matr
    # matr_resize = (matr_resize - matr_resize.min()) / (matr_resize.max() - matr_resize.min())
    
    #print('check processed matrix shape: ', matr_resize.shape)
    return matr_resize

  def build(self, discr_shape=(40,40), module_option=False, nb_samples=None):
    self.discr_shape = discr_shape
    
    if nb_samples is None:
        paths_i = glob.glob(self.global_path_i)
        print('check paths_i: ', paths_i[:5])
        paths_q = glob.glob(self.global_path_q)
    else:
        paths_i = glob.glob(self.global_path_i)[:nb_samples]
        paths_q = glob.glob(self.global_path_q)[:nb_samples]
    
    for path_i, path_q in zip(paths_i, paths_q):
        img_i =  self.__build_matr__(path_i)
        img_q =  self.__build_matr__(path_q)
        
        if module_option:
            img = (img_i**2 + img_q**2)[...,None]
            img = (img - img.min()) / (img.max() - img.min())
        else:
            #print('check scsling on module')
            img = np.sqrt(img_i**2 + img_q**2)
            
            img_i = (img_i - img.min()) / (img.max() - img.min())
            img_q = (img_q - img.min()) / (img.max() - img.min())
            
            img_i = img_i[...,None]
            img_q = img_q[...,None]
            img = np.concatenate((img_i, img_q), axis=2)
        
        self.data_samples.append({'table': img, 'label':self.label})
    return np.array(self.data_samples)
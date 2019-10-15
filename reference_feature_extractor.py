import numpy as np
from scipy.signal import argrelextrema
#import matplotlib.pyplot as plt

class FeatureExtractor():
    
    def __init__(self, matrix):
        self.matrix = matrix
        # get indices of matrix maximum
        self.ind_max = np.unravel_index(np.argmax(abs(self.matrix), axis=None), self.matrix.shape)
        # get 1D array on dopp max
        self.array = self.matrix[self.ind_max[0],:]
        
        #plt.plot(self.array)
        
        #print("CEHCK REF EXTRACTOR SHAPES: ", self.matrix.shape, self.array.shape, self.ind_max)
        
    def extract_f2(self):
        # get the number of local maxima
#        if self.matrix[self.ind_max] > 0:
#            print("CHECK POSITIVE PICK")
#            n_max = len(argrelextrema(self.array, np.greater)[0])
#        else:
#            print("CHECK NEGATIVE PICK")
#            self.array += 10
#            n_max = len(argrelextrema(self.array, np.greater)[0])
#        return n_max
        #print("NO SIGN CHECK")
        return len(argrelextrema(self.array, np.greater)[0])
    
    def extract_f3(self, tau):
        # convert index into chips
        x_chip = np.linspace(tau[0], tau[1], len(self.array))
        tau_max = x_chip[self.ind_max[1]]
        return (tau_max - x_chip.mean())**2
        
    
def most_frequent(List):
  return max(set(List), key=List.count)

def preprocess_df(df):
    df_temp = df.copy(deep=True)
    labels = df_temp['label']
    df_temp = df_temp.drop(['label'], axis=1)
    
    # average samples with sliding window 
    instance_size = 10
    instance_nb = int(df_temp.shape[0]/instance_size)
    X = np.zeros((instance_nb, df_temp.shape[1]))
    y = np.zeros((instance_nb))
    
    for instance_idx in range(instance_nb):
        X[instance_idx,:] = df_temp.iloc[instance_idx*instance_size : (instance_idx+1)*instance_size].as_matrix().mean(axis=0)
        y[instance_idx] = int(most_frequent(list(labels.iloc[instance_idx*instance_size : (instance_idx+1)*instance_size].as_matrix())))
    
    y = np.expand_dims(y, axis=1).astype('int64')
    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return X, y
        
        
        
        
        

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import numpy as np
#import tensorflow as tf
#from keras.utils import to_categorical
#from tqdm import tqdm, tqdm_notebook                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
import pandas as pd

#from skimage import filters

import json
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

#%% Import modules
from data_generator import CorrDatasetV2
from reference_feature_extractor import preprocess_df
# Verify that tensorflow is running with GPU
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# check if keras is using GPU
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
K.set_session(sess)

#%%
def main():
    #% Define const
    discr_size_fd = 13
    scale_code = 13
    Tint = 20e-3
    w = 10**6 # correlator bandwidth

	#% Read config files
    configs = []
    allFiles = glob.glob("config_ti20_combin/config_*.json") #config_dopp_ph0.json 
    for file_ in allFiles:
        with open(file_) as json_config_file:
            configs.append(json.load(json_config_file))
	
    print(configs[0])

    #% Create dataset for given config
    for config in configs:
        tau = [0, 2]
        dopp = [-2500, 2500]

        delta_tau = [config['delta_tau_min'], config['delta_tau_max']]
        delta_dopp = [config['delta_dopp_min'], config['delta_dopp_max']]
        alpha_att = [config['alpha_att_min'], config['alpha_att_max']]
        delta_phase = config['delta_phase'] * np.pi / 180
        cn0_logs = config['cn0_log']
                    
        for cn0_log in cn0_logs:
            for test_iter in range(20):
                for multipath_option in [True, False]:
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
        								    cn0_log=cn0_log, w=w)
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
        								    cn0_log=cn0_log, w=w)
        
                    if multipath_option:
                        samples, ref_data_samples_mp, module, delta_doppi, delta_taui, alpha_atti = Dataset.build(nb_samples=1000, ref_features=True)   
                    else:
                        samples, ref_data_samples_nomp, module = Dataset.build(nb_samples=1000, ref_features=True)   
                # create DF from ref_data_samples dict
                df_mp = pd.DataFrame(list(ref_data_samples_mp))
                df_nomp = pd.DataFrame(list(ref_data_samples_nomp))
                df = pd.concat([df_mp, df_nomp])
                
                X, y = preprocess_df(df)
                df_av = pd.DataFrame(X, columns=['f2', 'f3'])
                df_av['label'] = y
                features = ['f2', 'f3']
                target = ['label']
                
                # shuffle dataset
                df_av = df_av.sample(frac=1).reset_index(drop=True)
                
                df_train_val, df_test = train_test_split(df_av, test_size=0.2, random_state=42,
                                                         shuffle=True, stratify=df_av['label'])
                
                # Train, test split
                Xtrain_val, Xtest, ytrain_val, ytest = df_train_val[features], df_test[features], df_train_val[target], df_test[target] 
                Xtrain_val, Xtest, ytrain_val, ytest = Xtrain_val.values, Xtest.values, ytrain_val.values, ytest.values
                
                # define k-fold cross val / SVC
                kfold = KFold(n_splits=3, shuffle=True)
                model = SVC(kernel='rbf', gamma='auto', verbose=True)
                
                # train SVC
                scores = []
                for train_index, val_index in kfold.split(Xtrain_val):
                    # train_test split
                    Xtrain, Xval = Xtrain_val[train_index], Xtrain_val[val_index]
                    ytrain, yval = ytrain_val[train_index], ytrain_val[val_index]
                    
                    #scale features
                    scaler = StandardScaler()
                    scaler.fit(Xtrain)
                    Xtrain_norm = scaler.transform(Xtrain)
                    Xval_norm =scaler.transform(Xval)
                    
                    model.fit(Xtrain_norm, ytrain)
                    scores.append(model.score(Xval_norm, yval))
                
                # check preds / classification report
                Xtest_norm = scaler.transform(Xtest)
                preds = model.predict(Xtest_norm)
                
                precisions, recalls, _, __ = precision_recall_fscore_support(ytest.T[0], preds, average=None)
                precision, recall = precisions[1], recalls[1]
                accuracy = accuracy_score(ytest.T[0], preds)
                
                # write the logs to .json file
                logs = {'test_iter':test_iter,'config':config, 'val_acc':accuracy, 'val_precision': precision, 'val_recall': recall}
                if test_iter == 0:
                    with open('logs_ti20/logs_ref/output_cn0-{}_tau-{}-{}_dopp-{}-{}_phase-{}.json'.format(cn0_log, 
                										                  delta_tau[0], delta_tau[1],
                										                  delta_dopp[0], delta_dopp[1],
                												  config['delta_phase']), 'w') as outfile:
                        json.dump(logs, outfile)
                        outfile.write('\n')
                else:
                    with open('logs_ti20/logs_ref/output_cn0-{}_tau-{}-{}_dopp-{}-{}_phase-{}.json'.format(cn0_log, 
                										                  delta_tau[0], delta_tau[1],
                										                  delta_dopp[0], delta_dopp[1],
                												  config['delta_phase']), 'a') as outfile:
                        json.dump(logs, outfile)
                        outfile.write('\n')

                del model
                del Dataset
                del X, y, df_mp, df_nomp, df_av, df_train_val, df_test,  Xtrain_val, Xtest, ytrain_val, ytest, Xtrain, Xval, Xtrain_norm, Xval_norm, scores, preds
                del precisions, recalls, _, __, accuracy

#%%              
if __name__ == '__main__':
	main()

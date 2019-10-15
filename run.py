import sys

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if sys.argv[1] == '40':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	discr_size_fd = 40
	scale_code = 40
elif sys.argv[1] == '30':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	discr_size_fd = 30
	scale_code = 30
elif sys.argv[1] == '20':
	os.environ["CUDA_VISIBLE_DEVICES"] = "2"
	discr_size_fd = 20
	scale_code = 20
elif sys.argv[1] == '10':
	os.environ["CUDA_VISIBLE_DEVICES"] = "3"
	discr_size_fd = 10
	scale_code = 10
elif sys.argv[1] == '8':
	os.environ["CUDA_VISIBLE_DEVICES"] = "4"
	discr_size_fd = 8
	scale_code = 8
elif sys.argv[1] == '6':
	os.environ["CUDA_VISIBLE_DEVICES"] = "5"
	discr_size_fd = 6
	scale_code = 6
elif sys.argv[1] == '4':
	os.environ["CUDA_VISIBLE_DEVICES"] = "6"
	discr_size_fd = 4
	scale_code = 4

import numpy as np
from sklearn.model_selection import train_test_split

import json
import glob

from keras import optimizers
import keras_metrics

#%% Import modules
from data_generator import CorrDatasetV2
from model import Model, Model10, Model8, Model4

import tensorflow as tf
from keras import backend as K

# Verify that tensorflow is running with GPU
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# check if keras is using GPU
K.tensorflow_backend._get_available_gpus()
K.set_session(sess)


def main():
	#%% Define const
	Tint = 20e-3
	w = 10**6 # correlator bandwidth

	#%% Read config files
	
	configs = []
	allFiles = glob.glob("config_ti20/config_*.json") #config_dopp_ph0.json 
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

		print('CHECK JSON READ: ', delta_tau[0], delta_tau[1], delta_dopp[0], delta_dopp[1], alpha_att[0], alpha_att[1])

		for cn0_log in cn0_logs:
			for test_iter in range(20):
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
					dataset_temp = Dataset.build(nb_samples=1000)
					# Concatenate and shuffle arrays
					dataset = np.concatenate((dataset, dataset_temp[0]), axis=0)                
				

				# Split the data into train and val
				np.random.shuffle(dataset)
				data_train, data_val = train_test_split(dataset, test_size=0.2)

				X_train = np.array([x['table'] for x in data_train])
				X_val = np.array([x['table'] for x in data_val])

				y_train = np.array([x['label'] for x in data_train])
				y_val = np.array([x['label'] for x in data_val])


				# Define/ compile model
				if sys.argv[1] == '40':
					model = Model(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
				elif sys.argv[1] == '30':
					model = Model(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
				elif sys.argv[1] == '20':
					model = Model(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
				elif sys.argv[1] == '10':
					model = Model10(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
				elif sys.argv[1] == '8':
					model = Model8(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
				elif sys.argv[1] == '6':
					model = Model8(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
				elif sys.argv[1] == '4':
					model = Model4(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
				

				batch_size = 16
				train_iters = 10
				learning_rate = 1e-4
			    
				model.model.compile(loss='binary_crossentropy',
					  optimizer=optimizers.Adam(lr=learning_rate),
					  metrics=['acc',
						   keras_metrics.precision(),
						   keras_metrics.recall()])

				history = model.model.fit(
					  x=X_train,
					  y=y_train,
					  validation_data=(X_val, y_val),
					  epochs=train_iters,
					  batch_size=batch_size
				    )

				# write the logs to .json file
				history_dict = {k:[np.float64(i) for i in v] for k,v in history.history.items()}
				logs = {'test_iter':test_iter, 'config':config, 'history':history_dict}
				if test_iter == 0:
					with open('logs_ti20/logs{}/output_cn0-{}_tau-{}-{}_dopp-{}-{}_phase-{}.json'.format(
												discr_size_fd, cn0_log, 
										                delta_tau[0], delta_tau[1],
										               delta_dopp[0], delta_dopp[1],
											config['delta_phase']), 'w') as outfile:
					    json.dump(logs, outfile)
					    outfile.write('\n')
				else:
					with open('logs_ti20/logs{}/output_cn0-{}_tau-{}-{}_dopp-{}-{}_phase-{}.json'.format(
												discr_size_fd, cn0_log, 
										                delta_tau[0], delta_tau[1],
										               delta_dopp[0], delta_dopp[1],
											config['delta_phase']), 'a') as outfile: 
					    json.dump(logs, outfile)
					    outfile.write('\n')

				K.clear_session()
				del model
				del Dataset
				del dataset, dataset_temp, data_train, data_val, X_train, X_val, y_train, y_val, history

if __name__ == '__main__':
	main()
                    

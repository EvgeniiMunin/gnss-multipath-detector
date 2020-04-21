import numpy as np
import pandas as pd
from keras import layers
from keras import models
from keras import optimizers
import keras_metrics as km # for precsion/ recall metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint # to save best model
from keras.models import load_model

from sklearn.model_selection import train_test_split

import glob
import json
import matplotlib.pyplot as plt

# for openvino
import tensorflow as tf

#%%
from model import Model, Model10, Model8, Model4
from utils import save_model#, load_model

#%%

class IRModel:
    
    def __init__(self, h5_path, tf_path):
        self.keras_model_path = h5_path #'saved_models/best_model/best_mp_model.h5'
        self.tf_path = tf_path # 'tf_mp_model.pb'
        self.ir_path = ir_path
    
    def __freeze_session__(self, session, keep_var_names=None, output_names=None, clear_devices=True):
        '''freeze state of a session into a pruned compilation graph '''
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ''
            frozen_graph = convert_variables_to_constants(session, input_graph_def, 
                                                          output_names, freeze_var_names)
            return frozen_graph
        
    def __build_tf_graph__(self):
        '''keras to tf conversion'''
        # loading keras model
        K.set_learning_phase(0)
        model = load_model(self.keras_model_path,
                          custom_objects={
                              'binary_precision': km.binary_precision(),
                              'binary_recall': km.binary_recall()
                          })
        
        # create frozen graph of the keras model
        frozen_graph = self.__freeze_session__(K.get_session(), 
                                      output_names=[out.op.name for out in model.outputs])
        
        # save model as .pb file
        tf.train.write_graph(frozen_graph, 'saved_models/tf_model', self.tf_path, as_text=False)
            
        
    def build_ir_model(self, ir_shell_script):
        '''convert tensorflow graph to ir_model (from terminal or bash)'''
        ! chmod 744 convert_tf_ncs2.sh
        ! ./convert_tf_ncs2.sh
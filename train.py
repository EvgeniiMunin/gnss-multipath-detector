import keras_metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np

from data_generator_sx3 import SX3Dataset
from data_generator import CorrDatasetV2, FakeNoiseDataset
from utils import model_eval, load_ds_data

def train(model, X_train_synth, y_train_synth, X_val_sx, y_val_sx):
    """
        train and cross validate model
    """
    
    # prepare sx3 dataset
    dataset_nomp1 = SX3Dataset(label=0, global_path=data_path + 'sx_data/snap_no_mp_SX3_5_sat_11_89x81')
    dataset_nomp2 = SX3Dataset(label=0, global_path=data_path + 'sx_data/snap_no_mp_SX3_5_sat_18_89x81')
    dataset_mp1 = SX3Dataset(label=1, global_path=data_path + 'sx_data/snap_mp_SX3_5_sat_11_89x81')
    dataset_mp2 = SX3Dataset(label=1, global_path=data_path + 'sx_data/snap_mp_SX3_5_sat_18_89x81')
    data_nomp1 = dataset_nomp1.build(discr_shape=(80,80), nb_samples=100)
    data_nomp2 = dataset_nomp2.build(discr_shape=(80,80), nb_samples=100)
    data_mp1 = dataset_mp1.build(discr_shape=(80,80), nb_samples=100)
    data_mp2 = dataset_mp2.build(discr_shape=(80,80), nb_samples=100)
    
    data_val = np.concatenate((data_mp1, data_mp2, data_nomp1, data_nomp2), axis=0)
    np.random.shuffle(data_val)
    
    X_val_sx = np.array([x['table'] for x in data_val])
    y_val_sx = np.array([x['label'] for x in data_val])


    # prepare data generator data
    discr = 80
    X_train_synth, X_val_synth, y_train_synth, y_val_synth = load_ds_data(discr, data_path)

    # scale dataset
    X_train_synth = (X_train_synth - X_train_synth.mean()) / X_train_synth.std()
    X_val_sx = (X_val_sx - X_val_sx.mean()) / X_val_sx.std()

    # define model params    
    learning_rate = 1e-4
    optimizer = optimizers.Adam(lr=learning_rate)
    batch_size = 8
    train_iters = 20
    
    attn_model = model
    datagen = ImageDataGenerator(
              width_shift_range=0.2,
              height_shift_range=0.2,
              shear_range=0.01,
              zoom_range=[0.9,1.25],
              fill_mode='nearest',
              #zca_whitening=True,
              channel_shift_range=0.9,
              #brightness_range=[0.5,1.5]
              )
    datagen.fit(X_train_synth)
    
    attn_model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['acc',
                keras_metrics.precision(),
                keras_metrics.recall()])
    
    print(attn_model.summary())
    
    model_name = 'attn_model_dense_st-sc'
    
    checkpointer = ModelCheckpoint(filepath='{}.h5'.format(model_name), monitor='val_acc', verbose=1, save_best_only=True)
    #reduce_lr = LearningRateScheduler(lr_scheduler, verbose=1)
    
    history = attn_model.fit_generator(
            datagen.flow(X_train_synth, y_train_synth, batch_size=batch_size),
            validation_data=(X_val_sx, y_val_sx),
            epochs=train_iters,
            callbacks=[checkpointer]#, reduce_lr, tensorboard_callback]
            )
    
    attn_model.load_weights('{}.h5'.format(model_name))
    model_eval(attn_model, X_val_sx, y_val_sx, model_name, 0.5)
    
    return model
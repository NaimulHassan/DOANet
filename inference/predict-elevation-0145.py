from __future__ import print_function
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import Sequence
import h5py
import os

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, BatchNormalization, Input, Lambda, Activation, Conv2D, MaxPooling2D, Reshape, Bidirectional, TimeDistributed, GRU, GlobalMaxPooling1D
from keras.layers import concatenate
from keras.models import Model
from keras.utils import plot_model

#from model_lib import makeModel
from scipy.io import savemat

#%% Elevation prediction Model

def CNN_fibo(params):
    input_layer = Input(shape=(int(params['audio_len'] * params['sample_freq']), 8),
                        name='input_layer')

    # taking specific channels
    channels = [Lambda(lambda x: x[:, :, ch:ch + 1], name="channel_%d" % ch)(input_layer) for ch in params['channels']]
    extracted_channels = concatenate(channels, axis=-1)
    # taking specific window
    windowName = 'time_window_%1.2f_%1.2f' % (
        params['window_start'] / params['sample_freq'], params['window_end'] / params['sample_freq'])
    window = Lambda(lambda x: x[:, params['window_start']:params['window_end'], :, ], name=windowName)(
        extracted_channels)

    conv1 = Conv1D(filters=32, kernel_size=21, padding='valid', dilation_rate=1, activation='relu',
                   kernel_initializer='he_normal', name='conv1')(window)
    conv1 = BatchNormalization(name='conv1_bn')(conv1)
#    conv1 = MaxPooling1D(pool_size=2, padding='valid', name='conv1_maxpool')(conv1)

    conv2 = Conv1D(filters=64, kernel_size=19, padding='valid', dilation_rate=2, activation='relu',
                   kernel_initializer='he_normal', name='conv2')(conv1)
    conv2 = BatchNormalization(name='conv2_bn')(conv2)
    conv2 = MaxPooling1D(pool_size=3, padding='valid', name='conv2_maxpool')(conv2)

    conv3 = Conv1D(filters=128, kernel_size=17, padding='valid', dilation_rate=3, activation='relu',
                   kernel_initializer='he_normal', name='conv3')(conv2)
    conv3 = BatchNormalization(name='conv3_bn')(conv3)
#    conv3 = MaxPooling1D(pool_size=2, padding='valid', name='conv3_maxpool')(conv3)

    conv4 = Conv1D(filters=256, kernel_size=15, padding='valid', dilation_rate=5, activation='relu',
                   kernel_initializer='he_normal', name='conv4')(conv3)
    conv4 = BatchNormalization(name='conv4_bn')(conv4)
    conv4 = MaxPooling1D(pool_size=2, padding='valid', name='conv4_maxpool')(conv4)
    
    conv5 = Conv1D(filters=512, kernel_size=13, padding='valid', dilation_rate=8, activation='relu',
                   kernel_initializer='he_normal', name='conv5')(conv4)
    conv5 = BatchNormalization(name='conv5_bn')(conv5)
#    conv5 = MaxPooling1D(pool_size=2, padding='valid', name='conv5_maxpool')(conv5)
    
    conv6 = Conv1D(filters=1024, kernel_size=11, padding='valid', dilation_rate=13, activation='relu',
                   kernel_initializer='he_normal', name='conv6')(conv5)
    conv6 = BatchNormalization(name='conv6_bn')(conv6)
    conv6 = MaxPooling1D(pool_size=2, padding='valid', name='conv6_maxpool')(conv6)
    
    conv7 = Conv1D(filters=1024, kernel_size=9, padding='valid', dilation_rate=21, activation='relu',
                   kernel_initializer='he_normal', name='conv7')(conv6)
    conv7 = BatchNormalization(name='conv7_bn')(conv7)
#    conv7 = MaxPooling1D(pool_size=2, padding='valid', name='conv7_maxpool')(conv7)
    
    conv8 = Conv1D(filters=1024, kernel_size=9, padding='valid', dilation_rate=34, activation='relu',
                   kernel_initializer='he_normal', name='conv8')(conv7)
    conv8 = BatchNormalization(name='conv8_bn')(conv8)
    conv8 = MaxPooling1D(pool_size=2, padding='valid', name='conv8_maxpool')(conv8)
    
    conv9 = Conv1D(filters=1024, kernel_size=7, padding='valid', dilation_rate=55, activation='relu',
                   kernel_initializer='he_normal', name='conv9')(conv8)
    conv9 = BatchNormalization(name='conv9_bn')(conv9)

    max_pool = GlobalMaxPooling1D(data_format = 'channels_last')(conv9)

    flatten_block = Dropout(0.5, name='flatten_dropout')(max_pool)

    # azi_block = Dense(512, activation='relu', name='azi_dense1')(flatten_block)
    # azi_block = BatchNormalization(name='azi_dense1_bn')(azi_block)
    # azi_block = Dropout(0.5, name='dense_dropout')(azi_block)
    # azi_block = Dense(128, activation='relu', name='azi_dense2')(azi_block)
    # azi_block = BatchNormalization(name='azi_dense2_bn')(azi_block)
    # azi = Dense(2, activation='tanh', name='azi')(azi_block)

    elev_block = Dense(512, activation='relu', name='elev_dense1')(flatten_block)
    elev_block = BatchNormalization(name='elev_dense1_bn')(elev_block)
    # elev_block = Dropout(0.5, name='dense_dropout')(elev_block)
    # elev_block = Dense(128, activation='relu', name='elev_dense2')(elev_block)
    # elev_block = BatchNormalization(name='elev_dense2_bn')(elev_block)
    elev = Dense(1, name='elev')(elev_block)

    model = Model(inputs=input_layer, outputs=elev)

    return model

#%% Sample counter function
def sample_count(filepath):
    file = h5py.File(filepath, "r")
    keys = list(file.keys())
    return (np.shape(file[keys[-1]]))[0]
#%%Data Generator class

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, sample_size, data_source, batch_size=10, Fs = 44100, n_channel = 8,
                 shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.db = data_source
        self.Fs = 44100
        self.audio_len = int(Fs * 2.0)
        self.n_channel = n_channel
        self.list_IDs = np.arange(sample_size)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.sample_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        wav, xy, azimuth, elevation, index = self.__data_generation(list_IDs_temp)
        print('loaded files indexes are - ',index)
        return wav, xy, azimuth, elevation, index


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization        
        wav = np.empty((self.batch_size, self.audio_len, self.n_channel))
        x_cord = np.empty((self.batch_size))
        y_cord = np.empty((self.batch_size))
        azimuth = np.empty((self.batch_size))
        elevation = np.empty((self.batch_size))
        
        wav = self.db['wav'][list(np.sort(list_IDs_temp))[:]]
        x_cord = self.db['xcord'][list(np.sort(list_IDs_temp))[:]].ravel()
        y_cord = self.db['ycord'][list(np.sort(list_IDs_temp))[:]].ravel()
        azimuth = self.db['azimuth'][list(np.sort(list_IDs_temp))[:]].ravel()
        elevation = self.db['elevation'][list(np.sort(list_IDs_temp))[:]].ravel()

        return wav, np.swapaxes([x_cord, y_cord], 0, 1), azimuth, elevation, np.sort(list_IDs_temp)


#%% Frame parameters
windowStart = [0, 0.5, 1.0, 1.5]
windowEnd = [0.5, 1.0, 1.5, 2.0]

#%% test dataset
    
trained_model_path = "../input/static-task-models/best_elev_0145.hdf5"
val_filepath = "../input/dnn-prefilter-unet/static_task_train.hdf5"
db_val = h5py.File(val_filepath, 'r')
keys = list(db_val.keys())
val_sample_size = sample_count(val_filepath)
val_data_batch = 20
test_gen = DataGenerator(sample_size = val_sample_size, data_source = db_val, batch_size = val_data_batch)
predicted_azimuth_list = []
predicted_elevation_list = []
Actual_azimuth_list = []
Actual_elevation_list = []
Azimuth_error_list = []
Elevation_error_list = []
#%%Make model and load pretrained weights 
for i in np.arange(4):
    Fs = 44100
    params = {}
    params['audio_len'] = 2.0
    params['sample_freq'] = Fs
    params['channels'] = [0, 1, 4, 5]
    params['window_start'] = int(Fs * windowStart[i])
    params['window_end'] = int(Fs * windowEnd[i])
    
    #model = makeModel(modelArch, params)
    model = CNN_fibo(params)
    model.load_weights(trained_model_path)
    
    #%% Prediction
    predicted_elevation, Actual_elevation = [],[]
    for i in range(int(np.ceil(val_sample_size/val_data_batch))):
#    for i in range(2):
        wav, _, _, A_elevation, index = test_gen.__getitem__(i)
        p_elevation = model.predict(wav)
        Actual_elevation.extend(A_elevation)
        predicted_elevation.extend(p_elevation)
        del wav, A_elevation, p_elevation, index
        
    #%% Error calculation
        
    Elevation_err = np.asarray(np.abs(np.reshape(Actual_elevation, (-1,1)) - np.reshape(predicted_elevation, (-1,1)))*180.0, dtype = np.float64)
    np.place(Elevation_err, Elevation_err>180.0, 360.0 - Elevation_err[Elevation_err>180.0])
    
    Elevation_error_list.append(Elevation_err)
    predicted_elevation_list.append(predicted_elevation)
    Actual_elevation_list.append(Actual_elevation)
    del Elevation_err, predicted_elevation, Actual_elevation
    
#%% save predictions in a .mat file

pred_filepath = 'predictions_fibo_elevation_0145_noisy_competiton_train.mat'
savemat(pred_filepath, dict([#('static_azimuth', np.asarray(np.reshape(Actual_azimuth, (-1, 1))*180.0, dtype = np.float64)), 
                             ('static_elevation', np.swapaxes(np.asarray(Actual_elevation_list, dtype = np.float64)*180.0, 0, 1)),
                             #('predicted_static_azimuth', np.asarray(np.reshape(predicted_azimuth, (-1, 1))*180.0, dtype = np.float64)),
                             ('predicted_static_elevation', np.squeeze(np.swapaxes(np.asarray(predicted_elevation_list, dtype = np.float64)*180.0, 0, 1), axis = -1)),
                             #('azimuth_error', Azimuth_err),
                             ('elevation_error', np.swapaxes(np.squeeze(np.asarray(Elevation_error_list, dtype = np.float64), axis = -1), 0, 1))]))
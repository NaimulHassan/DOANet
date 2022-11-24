import os

import h5py
import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    MaxPooling1D,
    concatenate,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import Sequence
from pandas import read_csv


def CNN_fibo(params):
    input_layer = Input(
        shape=(int(params["audio_len"] * params["sample_freq"]), 8), name="input_layer"
    )

    # taking specific channels
    channels = [
        Lambda(lambda x: x[:, :, ch : ch + 1], name="channel_%d" % ch)(input_layer)
        for ch in params["channels"]
    ]
    extracted_channels = concatenate(channels, axis=-1)
    # taking specific window
    windowName = "time_window_%1.2f_%1.2f" % (
        params["window_start"] / params["sample_freq"],
        params["window_end"] / params["sample_freq"],
    )
    window = Lambda(
        lambda x: x[
            :,
            params["window_start"] : params["window_end"],
            :,
        ],
        name=windowName,
    )(extracted_channels)

    conv1 = Conv1D(
        filters=32,
        kernel_size=21,
        padding="valid",
        dilation_rate=1,
        activation="relu",
        kernel_initializer="he_normal",
        name="conv1",
    )(window)
    conv1 = BatchNormalization(name="conv1_bn")(conv1)

    conv2 = Conv1D(
        filters=64,
        kernel_size=19,
        padding="valid",
        dilation_rate=2,
        activation="relu",
        kernel_initializer="he_normal",
        name="conv2",
    )(conv1)
    conv2 = BatchNormalization(name="conv2_bn")(conv2)
    conv2 = MaxPooling1D(pool_size=3, padding="valid", name="conv2_maxpool")(conv2)

    conv3 = Conv1D(
        filters=128,
        kernel_size=17,
        padding="valid",
        dilation_rate=3,
        activation="relu",
        kernel_initializer="he_normal",
        name="conv3",
    )(conv2)
    conv3 = BatchNormalization(name="conv3_bn")(conv3)

    conv4 = Conv1D(
        filters=256,
        kernel_size=15,
        padding="valid",
        dilation_rate=5,
        activation="relu",
        kernel_initializer="he_normal",
        name="conv4",
    )(conv3)
    conv4 = BatchNormalization(name="conv4_bn")(conv4)
    conv4 = MaxPooling1D(pool_size=2, padding="valid", name="conv4_maxpool")(conv4)

    conv5 = Conv1D(
        filters=512,
        kernel_size=13,
        padding="valid",
        dilation_rate=8,
        activation="relu",
        kernel_initializer="he_normal",
        name="conv5",
    )(conv4)
    conv5 = BatchNormalization(name="conv5_bn")(conv5)

    conv6 = Conv1D(
        filters=1024,
        kernel_size=11,
        padding="valid",
        dilation_rate=13,
        activation="relu",
        kernel_initializer="he_normal",
        name="conv6",
    )(conv5)
    conv6 = BatchNormalization(name="conv6_bn")(conv6)
    conv6 = MaxPooling1D(pool_size=2, padding="valid", name="conv6_maxpool")(conv6)

    conv7 = Conv1D(
        filters=1024,
        kernel_size=9,
        padding="valid",
        dilation_rate=21,
        activation="relu",
        kernel_initializer="he_normal",
        name="conv7",
    )(conv6)
    conv7 = BatchNormalization(name="conv7_bn")(conv7)

    conv8 = Conv1D(
        filters=1024,
        kernel_size=9,
        padding="valid",
        dilation_rate=34,
        activation="relu",
        kernel_initializer="he_normal",
        name="conv8",
    )(conv7)
    conv8 = BatchNormalization(name="conv8_bn")(conv8)
    conv8 = MaxPooling1D(pool_size=2, padding="valid", name="conv8_maxpool")(conv8)

    conv9 = Conv1D(
        filters=1024,
        kernel_size=7,
        padding="valid",
        dilation_rate=55,
        activation="relu",
        kernel_initializer="he_normal",
        name="conv9",
    )(conv8)
    conv9 = BatchNormalization(name="conv9_bn")(conv9)

    max_pool = GlobalMaxPooling1D(data_format="channels_last")(conv9)

    flatten_block = Dropout(0.5, name="flatten_dropout")(max_pool)

    elev_block = Dense(512, activation="relu", name="elev_dense1")(flatten_block)
    elev_block = BatchNormalization(name="elev_dense1_bn")(elev_block)
    elev = Dense(1, name="elev")(elev_block)

    model = Model(inputs=input_layer, outputs=elev)

    return model


def sample_count(filepath):
    file = h5py.File(filepath, "r")
    keys = list(file.keys())
    return (np.shape(file[keys[-1]]))[0]


class DataGenerator(Sequence):
    "Generates data for Keras"

    def __init__(
        self, sample_size, data_source, batch_size=10, Fs=44100, n_channel=8, shuffle=False
    ):
        "Initialization"
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
        "Denotes the number of batches per epoch"
        return int(np.ceil(self.sample_size / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        wav, xy, azimuth, elevation, index = self.__data_generation(list_IDs_temp)
        return wav, elevation

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization

        wav = self.db["wav"][list(np.sort(list_IDs_temp))[:]]
        x_cord = self.db["xcord"][list(np.sort(list_IDs_temp))[:]].ravel()
        y_cord = self.db["ycord"][list(np.sort(list_IDs_temp))[:]].ravel()
        azimuth = self.db["azimuth"][list(np.sort(list_IDs_temp))[:]].ravel()
        elevation = self.db["elevation"][list(np.sort(list_IDs_temp))[:]].ravel()

        return wav, np.swapaxes([x_cord, y_cord], 0, 1), azimuth, elevation, np.sort(list_IDs_temp)


params = {}
params["audio_len"] = 2.0
params["sample_freq"] = 44100
params["channels"] = [2, 3, 6, 7]
params["window_start"] = int(44100 * 1.5)
params["window_end"] = int(44100 * 2)

model = CNN_fibo(params)
model.summary()

epochStart = 0
lr = 1e-3
opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None)
model.compile(optimizer=opt, loss="mse", metrics=[])

load_model = True
load_lr = False
load_epoch = False

pretrained_model_path = "../input/static-task-models/best_elev_2367.hdf5"
pretrained_logfile = "../input/static-task-models/trainingLog_2367_elev.csv"

if os.path.exists(pretrained_model_path) and load_model:
    print("pretrained Model Exists")
    model.load_weights(pretrained_model_path)
else:
    print('Pretrained Model doesn"t exist')

if load_lr or load_epoch:
    if os.path.exists(pretrained_logfile) and load_lr:
        csv_file = read_csv(pretrained_logfile)
        print("log_file_exists")
        lr = list(csv_file["lr"])[-1]
        print("lr = " + str(lr))
    if os.path.exists(pretrained_logfile) and load_epoch:
        csv_file = read_csv(pretrained_logfile)
        epochStart = list(csv_file["epoch"])[-1] + 1
        print("epochStart= " + str(epochStart))

train_filepath = "../input/dnn-prefilter-unet/static_task_train.hdf5"
val_filepath = "../input/dnn-prefilter-unet/static_task_val.hdf5"
db_train = h5py.File(train_filepath, "r")
db_val = h5py.File(val_filepath, "r")
keys = list(db_train.keys())
train_sample_size = sample_count(train_filepath)
val_sample_size = sample_count(val_filepath)
train_data_batch = 20
val_data_batch = 20

print("Training Sample Size = ", train_sample_size)
print("Validation Sample Size = ", val_sample_size)

train_gen = DataGenerator(
    sample_size=train_sample_size, data_source=db_train, batch_size=train_data_batch
)
val_gen = DataGenerator(sample_size=val_sample_size, data_source=db_val, batch_size=val_data_batch)

#  model path
modelWeightPath = "best_elev_2367.hdf5"
modelLogPath = "trainingLog_2367_elev.csv"

#  training callbacks
trainingCallbacks = []
trainingCallbacks.append(
    ModelCheckpoint(filepath=modelWeightPath, monitor="val_loss", verbose=0, save_best_only=True)
)
trainingCallbacks.append(
    EarlyStopping(monitor="val_loss", min_delta=1e-16, patience=25, verbose=0, mode="auto")
)
trainingCallbacks.append(
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.01,
        patience=5,
        verbose=1,
        min_lr=1e-15,
        mode="auto",
        min_delta=1e-16,
        cooldown=0,
    )
)
trainingCallbacks.append(CSVLogger(modelLogPath, separator=",", append=True))

#  starting training
model.fit_generator(
    generator=train_gen,
    validation_data=val_gen,
    epochs=100,
    verbose=1,
    initial_epoch=0,
    callbacks=trainingCallbacks,
)

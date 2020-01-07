import time
import datetime
import imageio
import numpy as np
from glob import glob
import os
from natsort import natsorted
import matplotlib.pyplot as plt
import rasterio
from osgeo import gdal
from rasterio.plot import show
import earthpy.plot as ep
import rasterio
from rasterio.enums import Resampling
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import glorot_uniform


# https://github.com/keras-team/keras/issues/10074
K.clear_session()

# Use the Sorensen-Dice coefficient as the metric, and the inverse of this as the loss function
# for training the model. See https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
def sorensen_dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef


def sorensen_dice_coef_loss(y_true, y_pred):
    return 1 - sorensen_dice_coef(y_true, y_pred)


def build_mangrove_model(input_shape):
    model_learning_rate = 0.00001
    model_initializer = glorot_uniform
    kernel_init = model_initializer()

    inputs = keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(conv4)
    drop4 = layers.Dropout(0.01)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(conv5)
    drop5 = layers.Dropout(0.01)(conv5)

    up6 = layers.concatenate(
        [layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', kernel_initializer=kernel_init)(drop5),
         drop4], axis=3)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(up6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(conv6)

    up7 = layers.concatenate(
        [layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer=kernel_init)(conv6),
         conv3], axis=3)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(up7)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(conv7)

    up8 = layers.concatenate(
        [layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=kernel_init)(conv7),
         conv2], axis=3)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(up8)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(conv8)

    up9 = layers.concatenate(
        [layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer=kernel_init)(conv8),
         conv1], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(up9)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(conv9)
    conv9 = layers.Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init)(conv9)

    conv10 = layers.Conv2D(2, (1, 1), activation='softmax')(conv9)

    model = keras.Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=model_learning_rate), loss=sorensen_dice_coef_loss, metrics=['binary_accuracy'])

    return model


def mangrove_model(test=False):

    trained_model_path = '/Users/cate/Documents/Data_Science_MSc/ECMM433/data/models/mangrove_unet_jean_luc.h5'

    jeanluc_trained = keras.models.load_model(trained_model_path,
                                              custom_objects={'sorensen_dice_coef_loss':sorensen_dice_coef_loss},
                                              compile=True)

    input_shape = (256, 256, 3)
    jeanluc_untrained = build_mangrove_model(input_shape=input_shape)

    # transfer weights to model with new input shape
    for i, layer in enumerate(jeanluc_untrained.layers):
        try:
            layer.set_weights(jeanluc_trained.layers[i].get_weights())

        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(jeanluc_untrained.to_json())

    if test:
        X = np.random.rand(1, 256, 256, 3)
        y_pred = jeanluc_untrained.predict(X)
        print('Untrained prediction: \n', y_pred)

        new_y_pred = new_model.predict(X)
        print ('New prediction: \n', new_y_pred)

    return new_model

if __name__== "__main__":

    veg_model = mangrove_model(test=True)

#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam


class VggLoss(object):

    def __init__(self, image_shape):

        self.image_shape = image_shape

    # computes VGG loss or content loss
    def model_loss(self, y_true, y_pred):

        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return K.mean(K.square(model(y_true) - model(y_pred)))


def get_optimizer():

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam


# Residual block
def res_block_gen(model, kernal_size, filters, strides):
    gen = model

    model = layers.Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = layers.BatchNormalization(momentum=0.5)(model)
    # Using Parametric ReLU
    model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = layers.Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = layers.BatchNormalization(momentum=0.5)(model)

    model = layers.add([gen, model])

    return model


def up_sampling_block(model, kernal_size, filters, strides):
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    # model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = layers.Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = layers.UpSampling2D(size=2)(model)
    model = layers.LeakyReLU(alpha=0.2)(model)

    return model


def discriminator_block(model, filters, kernel_size, strides):
    model = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = layers.BatchNormalization(momentum=0.5)(model)
    model = layers.LeakyReLU(alpha=0.2)(model)

    return model


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, noise_shape):

        self.noise_shape = noise_shape

    def generator(self):
        gen_input = keras.Input(shape=self.noise_shape)

        model = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
        model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)

        gen_model = model

        # Using 16 Residual Blocks
        for index in range(16):
            model = res_block_gen(model, 3, 64, 1)

        model = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
        model = layers.BatchNormalization(momentum=0.5)(model)
        model = layers.add([gen_model, model])

        # Using 2 UpSampling Blocks
        for index in range(2):
            model = up_sampling_block(model, 3, 256, 1)

        model = layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
        model = layers.Activation('tanh')(model)

        generator_model = keras.Model(inputs=gen_input, outputs=model)

        return generator_model


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Discriminator(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def discriminator(self):
        dis_input = keras.Input(shape=self.image_shape)

        model = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(dis_input)
        model = layers.LeakyReLU(alpha=0.2)(model)

        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        model = layers.Flatten()(model)
        model = layers.Dense(1024)(model)
        model = layers.LeakyReLU(alpha=0.2)(model)

        model = layers.Dense(1)(model)
        model = layers.Activation('sigmoid')(model)

        discriminator_model = keras.Model(inputs=dis_input, outputs=model)

        return discriminator_model

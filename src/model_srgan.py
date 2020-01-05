#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19



def residual_block(x):
    """
    Residual block
    """
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = layers.Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    res = layers.Activation(activation=activation)(res)
    res = layers.BatchNormalization(momentum=momentum)(res)

    res = layers.Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = layers.BatchNormalization(momentum=momentum)(res)

    # Add res and x
    res = layers.Add()([res, x])
    return res


def build_generator():
    residual_blocks = 16
    momentum = 0.8
    input_shape = (64, 64, 3)

    # Input Layer of the generator network
    input_layer = keras.Input(shape=input_shape, name='LR_img')

    # Add the pre-residual block
    gen1 = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)

    # Add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    # Add the post-residual block
    gen2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = layers.BatchNormalization(momentum=momentum)(gen2)

    # Take the sum of the output from the pre-residual block(gen1) and the post-residual block(gen2)
    gen3 = layers.Add()([gen2, gen1])

    # Add an upsampling block
    gen4 = layers.UpSampling2D(size=2)(gen3)
    gen4 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = layers.Activation('relu')(gen4)

    # Add another upsampling block
    gen5 = layers.UpSampling2D(size=2)(gen4)
    gen5 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = layers.Activation('relu')(gen5)

    # Output convolution layer
    gen6 = layers.Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = layers.Activation('tanh')(gen6)

    model = keras.Model(inputs=input_layer, outputs=output, name='generator')

    return model


def build_discriminator():
    """
    Create a discriminator network using the hyperparameter values defined below
    :return:
    """

    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (256, 256, 3)

    input_layer = keras.Input(shape=input_shape)

    # Add the first convolution block
    dis1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    # Add the 2nd convolution block
    dis2 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = layers.BatchNormalization(momentum=momentum)(dis2)
    dis2 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis2)

    # Add the third convolution block
    dis3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = layers.BatchNormalization(momentum=momentum)(dis3)
    dis3 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis3)

    # Add the fourth convolution block
    dis4 = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = layers.BatchNormalization(momentum=0.8)(dis4)
    dis4 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis4)

    # Add the fifth convolution block
    dis5 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = layers.BatchNormalization(momentum=momentum)(dis5)
    dis5 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis5)

    # Add the sixth convolution block
    dis6 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = layers.BatchNormalization(momentum=momentum)(dis6)
    dis6 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis6)

    # Add the seventh convolution block
    dis7 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = layers.BatchNormalization(momentum=momentum)(dis7)
    dis7 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis7)

    # Add the eight convolution block
    dis8 = layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = layers.BatchNormalization(momentum=momentum)(dis8)
    dis8 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis8)

    # Add a dense layer
    dis8 = layers.GlobalMaxPooling2D()(dis8)
    dis9 = layers.Dense(units=1024)(dis8)
    dis9 = layers.LeakyReLU(alpha=0.2)(dis9)

    # Last dense layer - for classification
    output = layers.Dense(units=1, activation='sigmoid')(dis9)

    model = keras.Model(inputs=[input_layer], outputs=output, name='discriminator')
    return model


def build_vgg():
    """
    Build VGG network to extract image features
    """
    input_shape = (256, 256, 3)

    # Load a pre-trained VGG19 model trained on 'Imagenet' dataset
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
    vgg.outputs = [vgg.layers[9].output]

    input_layer = keras.Input(shape=input_shape)

    # Extract features
    features = vgg(input_layer)

    # Create a Keras model
    model = keras.Model(inputs=[input_layer], outputs=[features])
    return model


def build_adversarial_model(generator, discriminator, vgg, low_resolution_shape, high_resolution_shape, common_optimizer):
    input_high_resolution = layers.Input(shape=high_resolution_shape)
    input_low_resolution = layers.Input(shape=low_resolution_shape)

    # Generate high-resolution images from low-resolution images
    generated_high_resolution_images = generator(input_low_resolution)

    # Extract feature maps of the generated images
    features = vgg(generated_high_resolution_images)

    # Make the discriminator network as trainable false
    discriminator.trainable = False

    # Get the probability of generated high-resolution images
    probs = discriminator(generated_high_resolution_images)

    # Create and compile an adversarial model combining
    adversarial_model = keras.Model([input_low_resolution, input_high_resolution], [probs, features])
    adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1],
                              optimizer=common_optimizer)

    return adversarial_model
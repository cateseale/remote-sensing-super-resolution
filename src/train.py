#!/usr/bin/env python3

import os
import numpy as np
import earthpy as ep
import tensorflow as tf
import matplotlib.pyplot as plt
from model_srgan import build_vgg, build_discriminator, build_generator, build_adversarial_model



def save_images(low_resolution_image, original_image, generated_image, path):
    """
    Save images in a single figure
    """
    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(1, 3, 1)
    ep.plot_rgb(np.moveaxis(low_resolution_image, -1, 0), ax=ax1, title='Low resolution image')

    ax2 = fig.add_subplot(1, 3, 2)
    ep.plot_rgb(np.moveaxis(original_image, -1, 0), ax=ax2, title='High resolution image')

    ax3 = fig.add_subplot(1, 3, 3)
    ep.plot_rgb(np.moveaxis(generated_image, -1, 0), ax=ax3, title='Super-resolution image')

    plt.savefig(path)


def build_networks():

    # Shape of low-resolution and high-resolution images
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)

    # Common optimizer for all networks
    common_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

    # Build and compile VGG19 network to extract features
    vgg = build_vgg()
    vgg.trainable = False
    vgg.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

    # Build the generator network
    generator = build_generator()

    # Build the adversarial network
    adversarial_model = build_adversarial_model(generator, discriminator, vgg, low_resolution_shape,
                                                high_resolution_shape, common_optimizer)

    return generator, discriminator, vgg, adversarial_model


def train(X_train, y_train, batch_size, epochs, export_path, export_sample_every=10):

    generator, discriminator, vgg, adversarial_model = build_networks()

    for epoch in range(epochs):
        print("Epoch:{}".format(epoch))

        """
        Train the discriminator network
        """

        # Sample a batch of images
        high_resolution_images, low_resolution_images = X_train, y_train

        # # Sample images and their conditioning counterparts
        # imgs_hr, imgs_lr = data_loader.load_data(batch_size)

        # Generate high resolution images
        generated_high_resolution_images = generator.predict(low_resolution_images)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator network on real and fake images
        d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        print("d_loss:", d_loss)

        """
        Train the generator network
        """

        # Extract features
        image_features = vgg.predict(high_resolution_images)

        # Train the generator network
        g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images],
                                                  [real_labels, image_features])

        print("g_loss:", g_loss)


        # Sample and save images after every x epochs
        if epoch % export_sample_every == 0:

            generated_images = generator.predict_on_batch(low_resolution_images)

            for index, img in enumerate(generated_images):
                save_images(low_resolution_images[index], high_resolution_images[index], img,
                            path= os.path.join(export_path, 'results', "/img_{}_{}".format(epoch,index)))


        # Save models
        generator.save_weights(os.path.join(export_path, 'models', '/generator.h5'))
        discriminator.save_weights(os.path.join(export_path, 'models', 'discriminator.h5'))


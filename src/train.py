#!/usr/bin/env python3

import os
import numpy as np
# from utils import plot_generated_images, save_images
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from preprocessing import DataLoader
from model import VggLoss, get_optimizer, Generator, Discriminator
from mangrove_model import MangroveLoss
from mlflow import log_metric, log_param, log_artifact


# Combined network
def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = keras.Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = keras.Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


def train(low_res_img_paths, high_res_img_paths, input_shape, output_shape, batch_size, epochs, data_dir, loss_model='vgg',
          workers=8):

    number_of_images = len(low_res_img_paths)
    number_of_batches = int(number_of_images / batch_size)

    model_save_dir = os.path.join(data_dir, 'models')
    output_dir = os.path.join(data_dir, 'results')
    log_param("model_save_dir", model_save_dir)
    log_param("output_dir", output_dir)

    if loss_model == 'vgg':
        print ('Using vgg loss')
        loss = VggLoss(output_shape)

    elif loss_model == 'mangrove':
        print ('using mangrove loss')
        loss = MangroveLoss(output_shape)

    generator = Generator(input_shape).generator()
    discriminator = Discriminator(output_shape).discriminator()

    optimizer = get_optimizer()
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    generator.compile(loss=loss.model_loss, optimizer=optimizer, options=run_opts)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = get_gan_network(discriminator, input_shape, generator, optimizer, loss.model_loss)

    loss_file = open(os.path.join(model_save_dir, 'losses.txt'), 'w+')
    loss_file.close()

    # Load batches of data
    enq = tf.keras.utils.OrderedEnqueuer(DataLoader(low_res_img_paths, high_res_img_paths, batch_size=batch_size))
    enq.start(workers=workers)
    gen = enq.get()

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)

        for i in tqdm(range(0, number_of_batches)):
            print('batch {}'.format(i))

            image_batch_lr, image_batch_hr = next(gen)
            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size * 6) - np.random.random_sample(batch_size * 6) * 0.2
            fake_data_Y = np.random.random_sample(batch_size * 6) * 0.2

            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            gan_Y = np.ones(batch_size * 6) - np.random.random_sample(batch_size * 6) * 0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])

            print("discriminator_loss : %f" % discriminator_loss)
            print("gan_loss :", gan_loss)

            # Log a metric; metrics can be updated throughout the run
            log_metric("gan_perceptual_loss", gan_loss[0])
            log_metric("gan_content_loss", gan_loss[1])
            log_metric("gan_adversarial_loss", gan_loss[2])
            log_metric("discriminator_loss", discriminator_loss)

            gan_loss = str(gan_loss)

            loss_output = os.path.join(model_save_dir, 'losses.txt')
            with open(loss_output, "w") as f:
                f.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' % (e, gan_loss, discriminator_loss))
            log_artifact(loss_output)

        if e % 1 == 0:
            generator.save(os.path.join(model_save_dir, 'gen_model%d.h5' % e))
            discriminator.save(os.path.join(model_save_dir, 'dis_model%d.h5' % e))




# def train(X_train, X_test, y_train, y_test, input_shape, output_shape, epochs, batch_size, data_dir, loss_model):
#
#
#
#     model_save_dir = os.path.join(data_dir, 'models')
#     output_dir = os.path.join(data_dir, 'results')
#
#     log_param("model_save_dir", model_save_dir)
#     log_param("output_dir", output_dir)
#
#     if loss_model == 'vgg':
#         print ('Using vgg loss')
#         loss = VggLoss(output_shape)
#
#     elif loss_model == 'mangrove':
#         print ('using mangrove loss')
#         loss = MangroveLoss(output_shape)
#
#     batch_count = int(X_train.shape[0] / batch_size)
#
#
#
#     generator = Generator(input_shape).generator()
#     discriminator = Discriminator(output_shape).discriminator()
#
#     optimizer = get_optimizer()
#     generator.compile(loss=loss.model_loss, optimizer=optimizer)
#     discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
#
#     gan = get_gan_network(discriminator, input_shape, generator, optimizer, loss.model_loss)
#
#     loss_file = open(os.path.join(model_save_dir, 'losses.txt'), 'w+')
#     loss_file.close()
#
#     for e in range(1, epochs + 1):
#         print('-' * 15, 'Epoch %d' % e, '-' * 15)
#         for _ in tqdm(range(batch_count)):
#             rand_nums = np.random.randint(0, y_train.shape[0], size=batch_size)
#
#             image_batch_hr = y_train[rand_nums]
#             image_batch_lr = X_train[rand_nums]
#             generated_images_sr = generator.predict(image_batch_lr)
#
#             real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
#             fake_data_Y = np.random.random_sample(batch_size) * 0.2
#
#             discriminator.trainable = True
#
#             d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
#             d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
#             discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
#
#             rand_nums = np.random.randint(0, y_train.shape[0], size=batch_size)
#             image_batch_hr = y_train[rand_nums]
#             image_batch_lr = X_train[rand_nums]
#
#             gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
#             discriminator.trainable = False
#             gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])
#
#             print("discriminator_loss : %f" % discriminator_loss)
#             print("gan_loss :", gan_loss)
#
#             # Log a metric; metrics can be updated throughout the run
#             log_metric("gan_perceptual_loss", gan_loss[0])
#             log_metric("gan_content_loss", gan_loss[1])
#             log_metric("gan_adversarial_loss", gan_loss[2])
#             log_metric("discriminator_loss", discriminator_loss)
#
#             gan_loss = str(gan_loss)
#
#             loss_output = os.path.join(model_save_dir, 'losses.txt')
#             with open(loss_output, "w") as f:
#                 f.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' % (e, gan_loss, discriminator_loss))
#             log_artifact(loss_output)
#
#         # if e == 1 or e % 5 == 0:
#         #     # plot_generated_images(output_dir, e, generator, y_test, X_test)
#         #     save_images(X_test, y_test, generator, path=os.path.join(output_dir, "img_epoch{}".format(e)))
#         if e % 200 == 0:
#             generator.save(os.path.join(model_save_dir, 'gen_model%d.h5' % e))
#             discriminator.save(os.path.join(model_save_dir, 'dis_model%d.h5' % e))

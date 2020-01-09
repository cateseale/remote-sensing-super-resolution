#!/usr/bin/env python3

import os
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import earthpy.plot as ep
from datetime import datetime

plt.switch_backend('agg')


def normalize(input_data):

    # return (input_data.astype(np.float32) - 127.5)/127.5
    return input_data


def denormalize(input_data):
    # input_data = (input_data + 1) * 127.5
    # return input_data.astype(np.uint8)
    return input_data


def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr, dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    print(examples)
    value = randint(0, examples)

    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    plt.figure(figsize=figsize)

    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)

    # plt.show()



def save_images(low_resolution_image, original_image, generator, path):
    """
    Save images in a single figure
    """

    examples = original_image.shape[0]
    print(examples)
    value = randint(0, examples)

    image_batch_hr = denormalize(original_image)
    image_batch_lr = low_resolution_image
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(1, 3, 1)
    ep.plot_rgb(np.moveaxis(image_batch_lr[value], -1, 0), ax=ax1, title='Low resolution image')

    ax2 = fig.add_subplot(1, 3, 2)
    ep.plot_rgb(np.moveaxis(image_batch_hr[value], -1, 0), ax=ax2, title='High resolution image')

    ax3 = fig.add_subplot(1, 3, 3)
    ep.plot_rgb(np.moveaxis(generated_image[value], -1, 0), ax=ax3, title='Super-resolution image')

    plt.savefig(path)


def save_train_test_split(image_splits_list, save_dir):

    # NOTE: image splits must be in this order  -  X_train, X_val, X_test, y_train, y_val, y_test

    timestamp_obj = datetime.now()
    data_created_timestamp = timestamp_obj.strftime('%Y%m%d_%H%M%S')

    filenames = [os.path.join(save_dir, 'images', data_created_timestamp +'_X_train'),
                 os.path.join(save_dir, 'images', data_created_timestamp + '_X_val'),
                 os.path.join(save_dir, 'images', data_created_timestamp + '_X_test'),
                 os.path.join(save_dir, 'images', data_created_timestamp + '_y_train'),
                 os.path.join(save_dir, 'images', data_created_timestamp + '_y_val'),
                 os.path.join(save_dir, 'images', data_created_timestamp + '_y_test')]

    for i in range(0,6):
        np.save(filenames[i], image_splits_list[i])

    print('Train/val/test image split created at {}'.format(data_created_timestamp))





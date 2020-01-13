#!/usr/bin/env python3

import os
import tensorflow as tf
from train import train
from preprocessing import list_images
from tensorflow.python.framework.ops import disable_eager_execution
from mlflow import log_metric, log_param, log_artifact


def run(path_to_data_folder, low_res_shape, high_res_shape, epochs, batch_size, loss_model='vgg'):

    disable_eager_execution()

    log_param("loss_model", loss_model)

    low_res_imgs_paths, high_res_imgs_paths = list_images(os.path.join(path_to_data_folder))


    train(low_res_imgs_paths, high_res_imgs_paths, low_res_shape, high_res_shape, batch_size, epochs, data_dir,
              loss_model=loss_model, workers=1)

    print ('Pipeline complete.')


if __name__== "__main__":

    epochs = 1
    batch_size = 64
    lr_shape = (64, 64, 3)
    hr_shape = (256, 256, 3)
    # data_dir = '/Users/cate/data/gans'
    data_dir = '/home/ec2-user/gan/data'


    log_param("epochs", epochs)
    log_param("batch_size", batch_size)
    log_param("lr_shape", lr_shape)
    log_param("hr_shape", hr_shape)
    log_param("data_dir", data_dir)

    run(data_dir, lr_shape, hr_shape, epochs, batch_size, loss_model='vgg')
    # run(data_dir, lr_shape, hr_shape, epochs, batch_size, loss_model='mangrove')

#!/usr/bin/env python3

from preprocessing import load_images
from train import train

from tensorflow.python.framework.ops import disable_eager_execution


def run(path_to_data_folder, low_res_shape, high_res_shape, epochs, batch_size, loss_model='vgg', continue_learning=False,
        pretrained_gen='', pretrained_dis=''):

    X_train, X_val, X_test, y_train, y_val, y_test = load_images(path_to_data_folder)

    if loss_model == 'mangrove':
        disable_eager_execution()

    if continue_learning:
        print('Using pretrained models {} and {}'.format(pretrained_gen, pretrained_dis))
        train(X_train, X_val, y_train, y_val, low_res_shape, high_res_shape, epochs, batch_size, path_to_data_folder,
              loss_model, pretrained_gen, pretrained_dis)

    else:
        print('Training new model')
        train(X_train, X_val, y_train, y_val, low_res_shape, high_res_shape, epochs, batch_size, path_to_data_folder, loss_model)

    print ('Pipeline complete.')


if __name__== "__main__":

    epochs = 2
    batch_size = 20
    lr_shape = (64, 64, 3)
    hr_shape = (256, 256, 3)
    data_dir = '/Users/cate/Documents/Data_Science_MSc/ECMM433/data'
    gen = 'gen_model1.h5'
    dis = 'dis_model1.h5'

    run(data_dir, lr_shape, hr_shape, epochs, batch_size, loss_model='vgg', continue_learning=False,
        pretrained_gen=gen, pretrained_dis=dis)
    # run(data_dir, lr_shape, hr_shape, epochs, batch_size, loss_model='mangrove')

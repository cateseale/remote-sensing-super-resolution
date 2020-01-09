#!/usr/bin/env python3


from train import train
from preprocessing import load_images
from utils import save_train_test_split
from tensorflow.python.framework.ops import disable_eager_execution
from mlflow import log_metric, log_param, log_artifact
import mlflow



def run(path_to_data_folder, low_res_shape, high_res_shape, epochs, batch_size, loss_model='vgg'):

    log_param("loss_model", loss_model)

    X_train, X_val, X_test, y_train, y_val, y_test = load_images(path_to_data_folder)

    save_train_test_split([X_train, X_val, X_test, y_train, y_val, y_test], data_dir)

    if loss_model == 'mangrove':
        disable_eager_execution()

    else:
        print('Training new model')
        train(X_train, X_val, y_train, y_val, low_res_shape, high_res_shape, epochs, batch_size, path_to_data_folder, loss_model)

    print ('Pipeline complete.')


if __name__== "__main__":

    epochs = 3000
    batch_size = 28
    lr_shape = (64, 64, 3)
    hr_shape = (256, 256, 3)
    data_dir = '/Users/cate/Documents/Data_Science_MSc/ECMM433/data'


    log_param("epochs", epochs)
    log_param("batch_size", batch_size)
    log_param("lr_shape", lr_shape)
    log_param("hr_shape", hr_shape)
    log_param("data_dir", data_dir)

    run(data_dir, lr_shape, hr_shape, epochs, batch_size, loss_model='vgg')
    # run(data_dir, lr_shape, hr_shape, epochs, batch_size, loss_model='mangrove')

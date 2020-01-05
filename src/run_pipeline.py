#!/usr/bin/env python3

from preprocessing import load_images
from train import train

def run(path_to_data_folder, batch_size, epochs):

    X_train, X_val, X_test, y_train, y_val, y_test = load_images(path_to_data_folder)

    train(X_train, y_train, batch_size, epochs, path_to_data_folder, export_sample_every=10)




if __name__ == "__main__":

    data_path = '/Users/cate/Documents/Data_Science_MSc/ECMM433/data/'

    batch_size = 10
    epochs = 30


    run(data_path, batch_size, epochs)
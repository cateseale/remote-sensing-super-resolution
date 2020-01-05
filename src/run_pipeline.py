#!/usr/bin/env python3

from preprocessing import load_images


def run(path_to_data_folder):

    X_train, X_val, X_test, y_train, y_val, y_test = load_images(path_to_data_folder)


if __name__ == "__main__":

    data_path = '/Users/cate/Documents/Data_Science_MSc/ECMM433/data/'

    run(data_path)
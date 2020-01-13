#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.switch_backend('agg')

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


def save_test_data(test_list, save_dir):

    timestamp_obj = datetime.now()
    data_created_timestamp = timestamp_obj.strftime('%Y%m%d_%H%M%S')

    filenames = [os.path.join(save_dir, 'images', data_created_timestamp + '_X_test'),
                 os.path.join(save_dir, 'images', data_created_timestamp + '_y_test')]

    for i in range(0,2):
        np.save(filenames[i], test_list[i])

    print('Test data created at {}'.format(data_created_timestamp))




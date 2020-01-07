#!/usr/bin/env python3

import os
import rasterio as rio
import numpy as np
from glob import glob
from natsort import natsorted
from rasterio.enums import Resampling
from skimage.exposure import rescale_intensity
from sklearn.model_selection import train_test_split


def _split_train_test_val(high_resolution_images, low_resolution_images):

    X_train, X_test, y_train, y_test = train_test_split(high_resolution_images, low_resolution_images, test_size=0.2,
                                                        random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print('Number of training images: {}\nNumber of validation images: {}\nNumber of test images: {}'.format(
        X_train.shape[0], X_val.shape[0], X_test.shape[0]))

    return X_train, X_val, X_test, y_train, y_val, y_test


def _rgb_from_bgr(image_arr):

    row, col, _ = image_arr.shape
    rgb_image = np.zeros((row, col, 3), dtype=np.float)

    p2, p98 = np.percentile(image_arr[:, :, 2], (2, 98))
    rgb_image[:, :, 0] = rescale_intensity(image_arr[:, :, 2], in_range=(p2, p98))

    p2, p98 = np.percentile(image_arr[:, :, 1], (2, 98))
    rgb_image[:, :, 1] = rescale_intensity(image_arr[:, :, 1], in_range=(p2, p98))

    p2, p98 = np.percentile(image_arr[:, :, 0], (2, 98))
    rgb_image[:, :, 2] = rescale_intensity(image_arr[:, :, 0], in_range=(p2, p98))

    return rgb_image


def _load_image_rgb(image_path, resample_img=False, scale_factor=0.25):

    if resample_img:

        with rio.open(image_path) as src:

            data = src.read(out_shape=(src.count,
                                       int(src.width * scale_factor),
                                       int(src.height * scale_factor)
                                       ),
                            resampling=Resampling.nearest
                            )

        channels_last = np.moveaxis(data[0:3, :, :], 0, -1)
        rgb = _rgb_from_bgr(channels_last)

        return rgb

    else:
        with rio.open(image_path) as src:
            data = src.read()

        channels_last = np.moveaxis(data[0:3, :, :], 0, -1)
        rgb = _rgb_from_bgr(channels_last)

        return rgb


def _data_loader(hr_paths, lr_paths):

    hr_images = []

    for item in hr_paths:
        image = _load_image_rgb(item)
        hr_images.append(image)

    lr_images = []

    for item in lr_paths:
        image = _load_image_rgb(item, resample_img=True, scale_factor=0.25)
        lr_images.append(image)

    hr_images = np.array(hr_images)
    lr_images = np.array(lr_images)

    return hr_images, lr_images


def _list_images(data_dir):

    high_resolution_images_dir = os.path.join(data_dir, 'images', 'high')
    low_resolution_images_dir = os.path.join(data_dir, 'images', 'low')

    high_resolution_images_list = natsorted(glob(os.path.join(high_resolution_images_dir, '*.tif')))
    low_resolution_images_list = natsorted(glob(os.path.join(low_resolution_images_dir, '*.tif')))

    no_of_hr_images = len(high_resolution_images_list)
    no_of_lr_images = len(low_resolution_images_list)

    print('Number of high resolution images: {}\nNumber of low resolution images: {}'.format(no_of_hr_images,
                                                                                             no_of_lr_images))

    assert no_of_hr_images == no_of_lr_images, 'Mismatch between the number of high and low resolution image pairs.'

    return high_resolution_images_list, low_resolution_images_list


def load_images(image_data_path):

    high_res_imgs_paths, low_res_imgs_paths = _list_images(image_data_path)

    high_res_imgs, low_res_imgs = _data_loader(high_res_imgs_paths, low_res_imgs_paths)

    hr_train, hr_val, hr_test, lr_train, lr_val, lr_test = _split_train_test_val(high_res_imgs, low_res_imgs)

    return lr_train, lr_val, lr_test, hr_train, hr_val, hr_test
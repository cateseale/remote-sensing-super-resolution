#!/usr/bin/env python3

import os
import math
import dask
import numpy as np
import rasterio as rio
import tensorflow as tf
from glob import glob
from dask import delayed
from natsort import natsorted
from rasterio.enums import Resampling
from skimage.exposure import rescale_intensity
from mlflow import log_metric, log_param, log_artifact



def list_images(data_dir):

    high_resolution_images_dir = os.path.join(data_dir, 'images', 'train', 'high')
    low_resolution_images_dir = os.path.join(data_dir, 'images', 'train', 'low')

    high_resolution_images_list = natsorted(glob(os.path.join(high_resolution_images_dir, '*.tif')))
    low_resolution_images_list = natsorted(glob(os.path.join(low_resolution_images_dir, '*.tif')))

    no_of_hr_images = len(high_resolution_images_list)
    no_of_lr_images = len(low_resolution_images_list)

    print('Number of high resolution images: {}\nNumber of low resolution images: {}'.format(no_of_hr_images,
                                                                                             no_of_lr_images))

    log_param("no_of_hr_images", no_of_hr_images)
    log_param("no_of_lr_images", no_of_lr_images)

    assert no_of_hr_images == no_of_lr_images, 'Mismatch between the number of high and low resolution image pairs.'

    return low_resolution_images_list, high_resolution_images_list


def rgb_from_bgr(image_arr):
    row, col, _ = image_arr.shape
    rgb_image = np.zeros((row, col, 3), dtype=np.float)

    p2, p98 = np.percentile(image_arr[:, :, 2], (2, 98))
    rgb_image[:, :, 0] = rescale_intensity(image_arr[:, :, 2], in_range=(p2, p98))

    p2, p98 = np.percentile(image_arr[:, :, 1], (2, 98))
    rgb_image[:, :, 1] = rescale_intensity(image_arr[:, :, 1], in_range=(p2, p98))

    p2, p98 = np.percentile(image_arr[:, :, 0], (2, 98))
    rgb_image[:, :, 2] = rescale_intensity(image_arr[:, :, 0], in_range=(p2, p98))

    return rgb_image


def load_image_rgb(image_path, resample_img=False, scale_factor=0.25):
    if resample_img:

        with rio.open(image_path) as src:

            data = src.read(out_shape=(src.count,
                                       int(src.width * scale_factor),
                                       int(src.height * scale_factor)
                                       ),
                            resampling=Resampling.nearest
                            )

        channels_last = np.moveaxis(data[0:3, :, :], 0, -1)
        rgb = rgb_from_bgr(channels_last)

        return rgb

    else:
        with rio.open(image_path) as src:
            data = src.read()

        channels_last = np.moveaxis(data[0:3, :, :], 0, -1)
        rgb = rgb_from_bgr(channels_last)

        return rgb


def rot_90(image):
    rotated90 = np.rot90(image, k=1, axes=(0, 1))
    return rotated90


def rot_180(image):
    rotated180 = np.rot90(image, k=2, axes=(0, 1))
    return rotated180


def rot_270(image):
    rotated270 = np.rot90(image, k=3, axes=(0, 1))
    return rotated270


def flip(image):
    flipped = np.flip(image, axis=2)
    return flipped


def flipped_lr(image):
    flippedlr = np.fliplr(image)
    return flippedlr


def image_augmenter(image_path, resample_img=False):
    image_list = []

    if resample_img:

        original = delayed(load_image_rgb)(image_path, resample_img=True)

    else:

        original = delayed(load_image_rgb)(image_path)

    image_90 = delayed(rot_90)(original)
    image_180 = delayed(rot_180)(original)
    image_270 = delayed(rot_270)(original)
    image_flipped = delayed(flip)(original)
    image_flippedlr = delayed(flipped_lr)(original)

    image_list.append(original)
    image_list.append(image_90)
    image_list.append(image_180)
    image_list.append(image_270)
    image_list.append(image_flipped)
    image_list.append(image_flippedlr)

    image_list_computed = dask.compute(*image_list)

    image_arr = np.array(image_list_computed, dtype='float32')

    return image_arr


class DataLoader(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(np.concatenate([image_augmenter(file_name, resample_img=True) for file_name in batch_x])), \
               np.array(np.concatenate([image_augmenter(file_name) for file_name in batch_y]))

# def _split_train_test_val(high_resolution_images, low_resolution_images, test_size=0.2):
#
#     X_train, X_test, y_train, y_test = train_test_split(high_resolution_images, low_resolution_images, test_size=test_size,
#                                                         random_state=42)
#
#     X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
#
#     print('Number of training images: {}\nNumber of validation images: {}\nNumber of test images: {}'.format(
#         X_train.shape[0], X_val.shape[0], X_test.shape[0]))
#
#     log_param("Training image pairs", X_train.shape[0])
#     log_param("Validation image pairs", X_val.shape[0])
#     log_param("Test image pairs", X_test.shape[0])
#
#
#     return X_train, X_val, X_test, y_train, y_val, y_test
#
#
# def rot_90(image):
#     rotated90 = np.rot90(image, k=1, axes=(0, 1))
#     return rotated90
#
#
# def rot_180(image):
#     rotated180 = np.rot90(image, k=2, axes=(0, 1))
#     return rotated180
#
#
# def rot_270(image):
#     rotated270 = np.rot90(image, k=3, axes=(0,1))
#     return rotated270
#
#
# def flip(image):
#     flipped = np.flip(image, axis=2)
#     return flipped
#
#
# def flipped_lr(image):
#     flippedlr = np.fliplr(image)
#     return flippedlr
#
#
# def _rgb_from_bgr(image_arr):
#
#     row, col, _ = image_arr.shape
#     rgb_image = np.zeros((row, col, 3), dtype=np.float)
#
#     p2, p98 = np.percentile(image_arr[:, :, 2], (2, 98))
#     rgb_image[:, :, 0] = rescale_intensity(image_arr[:, :, 2], in_range=(p2, p98))
#
#     p2, p98 = np.percentile(image_arr[:, :, 1], (2, 98))
#     rgb_image[:, :, 1] = rescale_intensity(image_arr[:, :, 1], in_range=(p2, p98))
#
#     p2, p98 = np.percentile(image_arr[:, :, 0], (2, 98))
#     rgb_image[:, :, 2] = rescale_intensity(image_arr[:, :, 0], in_range=(p2, p98))
#
#     return rgb_image
#
#
# def _load_image_rgb(image_path, resample_img=False, scale_factor=0.25):
#
#     if resample_img:
#
#         with rio.open(image_path) as src:
#
#             data = src.read(out_shape=(src.count,
#                                        int(src.width * scale_factor),
#                                        int(src.height * scale_factor)
#                                        ),
#                             resampling=Resampling.nearest
#                             )
#
#         channels_last = np.moveaxis(data[0:3, :, :], 0, -1)
#         rgb = _rgb_from_bgr(channels_last)
#
#         return rgb
#
#     else:
#         with rio.open(image_path) as src:
#             data = src.read()
#
#         channels_last = np.moveaxis(data[0:3, :, :], 0, -1)
#         rgb = _rgb_from_bgr(channels_last)
#
#         return rgb
#
#
# def _data_loader(hr_paths, lr_paths):
#     hr_image_list = []
#     lr_image_list = []
#
#     for item in hr_paths:
#         original = delayed(_load_image_rgb)(item)
#         image_90 = delayed(rot_90)(original)
#         image_180 = delayed(rot_180)(original)
#         image_270 = delayed(rot_270)(original)
#         image_flipped = delayed(flip)(original)
#         image_flippedlr = delayed(flipped_lr)(original)
#
#         hr_image_list.append(original)
#         hr_image_list.append(image_90)
#         hr_image_list.append(image_180)
#         hr_image_list.append(image_270)
#         hr_image_list.append(image_flipped)
#         hr_image_list.append(image_flippedlr)
#
#     for item in lr_paths:
#         original = delayed(_load_image_rgb)(item, resample_img=True)
#         image_90 = delayed(rot_90)(original)
#         image_180 = delayed(rot_180)(original)
#         image_270 = delayed(rot_270)(original)
#         image_flipped = delayed(flip)(original)
#         image_flippedlr = delayed(flipped_lr)(original)
#
#         lr_image_list.append(original)
#         lr_image_list.append(image_90)
#         lr_image_list.append(image_180)
#         lr_image_list.append(image_270)
#         lr_image_list.append(image_flipped)
#         lr_image_list.append(image_flippedlr)
#
#     hr_image_list_computed = dask.compute(*hr_image_list)
#     lr_image_list_computed = dask.compute(*lr_image_list)
#
#     hr_image_arr = np.array(hr_image_list_computed, dtype='float32')
#     lr_image_arr = np.array(lr_image_list_computed, dtype='float32')
#
#     return hr_image_arr, lr_image_arr
#
#
# def _list_images(data_dir):
#
#     high_resolution_images_dir = os.path.join(data_dir, 'images', 'high')
#     low_resolution_images_dir = os.path.join(data_dir, 'images', 'low')
#
#     high_resolution_images_list = natsorted(glob(os.path.join(high_resolution_images_dir, '*.tif')))
#     low_resolution_images_list = natsorted(glob(os.path.join(low_resolution_images_dir, '*.tif')))
#
#     no_of_hr_images = len(high_resolution_images_list)
#     no_of_lr_images = len(low_resolution_images_list)
#
#     print('Number of high resolution images: {}\nNumber of low resolution images: {}'.format(no_of_hr_images,
#                                                                                              no_of_lr_images))
#
#     log_param("no_of_hr_images", no_of_hr_images)
#     log_param("no_of_lr_images", no_of_lr_images)
#
#     assert no_of_hr_images == no_of_lr_images, 'Mismatch between the number of high and low resolution image pairs.'
#
#     return high_resolution_images_list, low_resolution_images_list
#
#
# def load_images(image_data_path):
#
#     high_res_imgs_paths, low_res_imgs_paths = _list_images(image_data_path)
#
#     high_res_imgs, low_res_imgs = _data_loader(high_res_imgs_paths, low_res_imgs_paths)
#
#     hr_train, hr_val, hr_test, lr_train, lr_val, lr_test = _split_train_test_val(high_res_imgs, low_res_imgs)
#
#     return lr_train, lr_val, lr_test, hr_train, hr_val, hr_test
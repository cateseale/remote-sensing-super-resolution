import tensorflow as tf
import os
from glob import glob
import numpy as np


def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img)
    return img


def calculate_metrics(high_res_dir, super_dir):
    img_list = glob(os.path.join(high_res_dir, '*.png'))
    image_ids = [os.path.basename(path)[4:-4] for path in img_list]

    psnrs = []
    ssims = []

    for img_id in image_ids:
        hr_img_path = os.path.join(high_res_dir, 'img_' + str(img_id) + '.png')
        sr_img_path = os.path.join(super_dir, 'img_' + str(img_id) + '.png')

        hr_img = decode_image(hr_img_path)
        sr_img = decode_image(sr_img_path)

        psnr_result = tf.image.psnr(hr_img, sr_img, max_val=255)
        ssim_result = tf.image.ssim(hr_img, sr_img, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

        psnrs.append(psnr_result)
        ssims.append(ssim_result)

    return np.mean(psnrs), np.mean(ssims)


if __name__ == "__main__":

    hr_images_dir = '/Users/cate/data/gans/images_rgb/test/high'
    # sr_images_dir = '/Users/cate/data/gans/results/test_images_pretrained_SRGAN/resolved_SRResNet'
    # sr_images_dir = '/Users/cate/data/gans/results/test_images_pretrained_SRGAN/resolved_SRGAN'
    sr_images_dir = '/Users/cate/data/gans/results/test_images_bicubic'

    psnr, ssim = calculate_metrics(hr_images_dir, sr_images_dir)

    print('PSNR: ', psnr)
    print('SSIM: ', ssim)
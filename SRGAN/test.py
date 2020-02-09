from model import resolve_single
from utils import load_image, save_image
from model.srgan import generator, discriminator
import matplotlib.pyplot as plt
import os
from glob import glob


def resolve_and_plot(lr_image_path, hr_image_path, save_dir, img_id):
    lr = load_image(lr_image_path)
    hr = load_image(hr_image_path)

    pre_sr = resolve_single(pre_generator, lr)
    gan_sr = resolve_single(gan_generator, lr)

    save_image(pre_sr, save_dir, 'resolved_SRResNet', str(img_id))
    save_image(gan_sr, save_dir, 'resolved_SRGAN', str(img_id))

    images = [lr, hr, pre_sr, gan_sr]
    titles = ['Low Resolution', 'High Resolution', 'Super Resolution (PRE)', 'Super Resolution (GAN)']
    positions = [1, 2, 3, 4]

    plt.figure(figsize=(20, 20))

    for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
        plt.subplot(2, 2, pos)
        # plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    plt.savefig(os.path.join(save_dir, 'comparison_plots', 'test_img_' + str(img_id) + '.png'))
    plt.close()


def plot_test_images(lr_dir, hr_dir, out_dir):
    lr_images = glob(os.path.join(lr_dir, '*.png'))
    hr_images = glob(os.path.join(hr_dir, '*.png'))

    if len(lr_images) != len(hr_images):
        raise ValueError

    no_of_images = len(lr_images)

    print(no_of_images)

    for i in range(0, no_of_images):
        resolve_and_plot(lr_images[i], hr_images[i], out_dir, i)


if __name__ == "__main__":

    pre_generator = generator()
    gan_generator = generator()

    # Location of model weights (needed for demo)
    # weights_dir = 'weights/srgan'
    # weights_file = lambda filename: os.path.join(weights_dir, filename)
    #
    # pre_generator.load_weights(weights_file('pre_generator.h5'))
    # gan_generator.load_weights(weights_file('gan_generator.h5'))

    # pre_generator.load_weights('/Users/cate/data/gans/aws/weights/srgan_gen_trained_for_100K_GAN_trained_for_100K/pre_generator.h5')
    # gan_generator.load_weights('/Users/cate/data/gans/aws/weights/srgan_gen_trained_for_100K_GAN_trained_for_100K/gan_generator.h5')

    pre_generator.load_weights('/Users/cate/git/remote-sensing-super-resolution/SRGAN/weights/srgan/pre_generator.h5')
    gan_generator.load_weights('/Users/cate/git/remote-sensing-super-resolution/SRGAN/weights/srgan/gan_generator.h5')

    # Point towards the images to test
    lr_image_dir = '/Users/cate/data/gans/images_rgb/test/low/'
    hr_image_dir = '/Users/cate/data/gans/images_rgb/test/high/'

    # lr_image_dir = '/Users/cate/data/gans/data_prep/houston_prep/delete/images_rgb_64x64'
    # hr_image_dir = '/Users/cate/data/gans/data_prep/houston_prep/delete/images_rgb_64x64'

    # Dir to save the results
    test_sve_dir = '/Users/cate/data/gans/results/test_no_training'
    # test_sve_dir = '/Users/cate/data/gans/results/test_images_pretrained_SRGAN'
    # test_sve_dir = '/Users/cate/data/gans/data_prep/houston/test_images_sr_pretrained_SRGAN'

    plot_test_images(lr_image_dir, hr_image_dir, test_sve_dir)

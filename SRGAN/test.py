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

    plt.figure(figsize=(20, 20))

    images = [lr, hr, pre_sr, gan_sr]
    titles = ['Low Resolution', 'High Resolution', 'Super Resolution (PRE)', 'Super Resolution (GAN)']
    positions = [1, 2, 3, 4]

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

    image_ids = [os.path.basename(path)[4:-4] for path in lr_images]

    for image_id in image_ids:
        lr_path = os.path.join(lr_dir, 'img_' + str(image_id) + '.png')
        hr_path = os.path.join(hr_dir, 'img_' + str(image_id) + '.png')

        resolve_and_plot(lr_path, hr_path, out_dir, image_id)


if __name__ == "__main__":

    pre_generator = generator()
    gan_generator = generator()

    # Location of model weights (needed for demo)
    weights_dir = 'weights/srgan'
    weights_file = lambda filename: os.path.join(weights_dir, filename)

    # pre_generator.load_weights(weights_file('pre_generator.h5'))
    # gan_generator.load_weights(weights_file('gan_generator.h5'))

    pre_generator.load_weights('/Users/cate/data/gans/aws/weights/srgan_gen_trained_for_100K_GAN_trained_for_100K/pre_generator.h5')
    gan_generator.load_weights('/Users/cate/data/gans/aws/weights/srgan_gen_trained_for_100K_GAN_trained_for_100K/gan_generator.h5')

    # Point towards the images to test
    lr_image_dir = '/Users/cate/data/gans/images_rgb/test/low/'
    hr_image_dir = '/Users/cate/data/gans/images_rgb/test/high/'

    # Dir to save the results
    test_sve_dir = '/Users/cate/data/gans/results/test_images_pretrained_SRGAN'

    plot_test_images(lr_image_dir, hr_image_dir, test_sve_dir)

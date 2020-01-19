from model import resolve_single
from utils import load_image
from model.srgan import generator, discriminator
import matplotlib.pyplot as plt
import os

def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)

    pre_sr = resolve_single(pre_generator, lr)
    gan_sr = resolve_single(gan_generator, lr)

    plt.figure(figsize=(20, 20))

    images = [lr, pre_sr, gan_sr]
    titles = ['LR', 'SR (PRE)', 'SR (GAN)']
    positions = [1, 3, 4]

    for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
        plt.subplot(2, 2, pos)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

        plt.savefig('test.png')



if __name__ == "__main__":

    pre_generator = generator()
    gan_generator = generator()

    # Location of model weights (needed for demo)
    weights_dir = 'weights/srgan'
    weights_file = lambda filename: os.path.join(weights_dir, filename)


    pre_generator.load_weights(weights_file('pre_generator.h5'))
    gan_generator.load_weights(weights_file('gan_generator.h5'))

    resolve_and_plot('/Users/cate/data/gans/images_rgb/val/low/img_165.png')

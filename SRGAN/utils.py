import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import os


def load_image(path):
    return np.array(Image.open(path))


def save_image(array, out_dir, folder, im_id):
    imageio.imwrite(os.path.join(out_dir, folder, 'img_' + im_id + '.png'), array)

def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

import os
import tensorflow as tf
from natsort import natsorted
from glob import glob
import functools
from tensorflow.python.data.experimental import AUTOTUNE


class CATESR:
    def __init__(self,
                 subset='train',
                 images_dir='/Users/cate/data/gans/images_rgb',
                 caches_dir='/Users/cate/data/gans/caches_rgb'):

        if subset == 'train':
            self.images_dir = os.path.join(images_dir, 'train')
            self.caches_dir = os.path.join(caches_dir, 'train')
        elif subset == 'valid':
            self.images_dir = os.path.join(images_dir, 'val')
            self.caches_dir = os.path.join(caches_dir, 'val')
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        self.subset = subset
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def image_ids(self):

        img_list_HR = natsorted(glob(os.path.join(self._hr_images_dir(), '*.png')))
        img_list_LR = natsorted(glob(os.path.join(self._lr_images_dir(), '*.png')))

        ids_HR = list(map(lambda sub: int(''.join([ele for ele in sub if ele.isnumeric()])), [i[30:] for i in img_list_HR]))
        ids_LR = list(map(lambda sub: int(''.join([ele for ele in sub if ele.isnumeric()])), [i[30:] for i in img_list_LR]))

        if functools.reduce(lambda i, j: i and j, map(lambda m, k: m == k, ids_HR, ids_LR), True):
            return ids_HR
        else:
            raise ValueError("HR images and LR images are not identical")

    def dataset(self, batch_size=16, repeat_count=None, random_transform=True, shuffle_buffer_size=1000):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        if random_transform:
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds


    def hr_dataset(self):

        ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file())

        if not os.path.exists(self._hr_cache_index()):
            self._populate_cache(ds, self._hr_cache_file())

        return ds

    def lr_dataset(self):

        ds = self._images_dataset(self._lr_image_files()).cache(self._lr_cache_file())

        if not os.path.exists(self._lr_cache_index()):
            self._populate_cache(ds, self._lr_cache_file())

        return ds

    def _hr_images_dir(self):
        return os.path.join(self.images_dir, 'high')

    def _lr_images_dir(self):
        return os.path.join(self.images_dir, 'low')

    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [os.path.join(images_dir, 'img_{0}.png'.format(image_id)) for image_id in self.image_ids()]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [os.path.join(images_dir, 'img_{0}.png'.format(image_id)) for image_id in self.image_ids()]

    def _hr_cache_file(self):
        return os.path.join(self.caches_dir, 'CATESR_{0}_HR.cache'.format(self.subset))

    def _lr_cache_file(self):
        return os.path.join(self.caches_dir, 'CATESR_{0}_LR.cache'.format(self.subset))

    def _hr_cache_index(self):
        return '{0}.index'.format(self._hr_cache_file())

    def _lr_cache_index(self):
        return '{0}.index'.format(self._lr_cache_file())

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print('Caching decoded images in {0} ...'.format(cache_file))
        for _ in ds: pass
        print('Cached decoded images in {0}.'.format(cache_file))



def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)



def download_archive(file, target_dir, extract=True):
    source_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/{0}'.format(file)
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file))

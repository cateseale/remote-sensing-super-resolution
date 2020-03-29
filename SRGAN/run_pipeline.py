import os
from data import CATESR
from model.srgan import generator, discriminator
from train import SrganTrainer, SrganGeneratorTrainer
import mlflow

if __name__ == "__main__":

    # Location of model weights (needed for demo)
    weights_dir = 'weights/srgan'
    weights_file = lambda filename: os.path.join(weights_dir, filename)

    os.makedirs(weights_dir, exist_ok=True)

    # catesr_train = CATESR(subset='train', images_dir='/Users/cate/data/gans/images_rgb',
    #                       caches_dir='/Users/cate/data/gans/caches_rgb')
    # catesr_valid = CATESR(subset='valid', images_dir='/Users/cate/data/gans/images_rgb',
    #                       caches_dir='/Users/cate/data/gans/caches_rgb')


    catesr_train = CATESR(subset='train', images_dir='/home/ec2-user/gan/data/images_rgb',
                          caches_dir='/home/ec2-user/gan/data/caches_rgb')
    catesr_valid = CATESR(subset='valid', images_dir='/home/ec2-user/gan/data/images_rgb',
                          caches_dir='/home/ec2-user/gan/data/caches_rgb')


    train_ds = catesr_train.dataset(batch_size=16, random_transform=True, shuffle_buffer_size=500)
    valid_ds = catesr_valid.dataset(batch_size=16, random_transform=True, shuffle_buffer_size=500)


    # First train the generator

    generator_model = generator()
    generator_model.load_weights(os.path.join(weights_dir, 'pretrained_gan_generator.h5'))

    pre_trainer = SrganGeneratorTrainer(model=generator_model, checkpoint_dir=f'.ckpt/pre_generator')
    pre_trainer.train(train_ds,
                      valid_ds.take(100),
                      steps=100000,
                      evaluate_every=1000,
                      save_best_only=True)

    pre_trainer.model.save_weights(weights_file('pre_generator.h5'))


    # Generator fine - tuning(GAN)
    gan_generator = generator()
    gan_generator.load_weights(weights_file('pre_generator.h5'))

    gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
    gan_trainer.train(train_ds, steps=200000)

    gan_trainer.generator.save_weights(weights_file('gan_generator.h5'))
    gan_trainer.discriminator.save_weights(weights_file('gan_discriminator.h5'))

    print('Training finished.')


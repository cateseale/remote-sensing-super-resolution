import os
from data import CATESR
from train import WdsrTrainer, SrganTrainer
from model.wdsr import wdsr_b
from model.srgan import discriminator


if __name__ == "__main__":

    depth = 32    # Number of residual blocks
    scale = 4    # Super-resolution factor

    # Location of model weights (needed for demo)
    weights_dir = f'weights/wdsr-{depth}-x{scale}'
    weights_file = os.path.join(weights_dir, 'weights.h5')

    os.makedirs(weights_dir, exist_ok=True)


    # catesr_train = CATESR(subset='train', images_dir='/Users/cate/data/gans/images_rgb',
    #                       caches_dir='/Users/cate/data/gans/caches_rgb')
    # catesr_valid = CATESR(subset='valid', images_dir='/Users/cate/data/gans/images_rgb',
    #                       caches_dir='/Users/cate/data/gans/caches_rgb')


    catesr_train = CATESR(subset='train', images_dir='/home/ec2-user/gan/data/images_rgb',
                          caches_dir='/home/ec2-user/gan/data/caches_rgb')
    catesr_valid = CATESR(subset='valid', images_dir='/home/ec2-user/gan/data/images_rgb',
                          caches_dir='/home/ec2-user/gan/data/caches_rgb')

    # train_ds = catesr_train.dataset(batch_size=16, random_transform=True, shuffle_buffer_size=500)
    # valid_ds = catesr_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

    # trainer = WdsrTrainer(model=wdsr_b(scale=scale, num_res_blocks=depth),
    #                       checkpoint_dir=f'.ckpt/wdsr-b-{depth}-x{scale}')
    #
    # # Train WDSR B model for 300,000 steps and evaluate model
    # # every 1000 steps on the first 10 images of the DIV2K
    # # validation set. Save a checkpoint only if evaluation
    # # PSNR has improved.
    # trainer.train(train_ds,
    #               valid_ds.take(10),
    #               steps=300000,
    #               evaluate_every=1000,
    #               save_best_only=True)
    #
    # # Restore from checkpoint with highest PSNR
    # trainer.restore()
    #
    # # Evaluate model on full validation set
    # psnr = trainer.evaluate(valid_ds)
    # print(f'PSNR = {psnr.numpy():3f}')
    #
    # # Save weights to separate location (needed for demo)
    # trainer.model.save_weights(weights_file)

    # Custom WDSR B model (0.62M parameters)
    generator = wdsr_b(scale=4, num_res_blocks=32)
    generator.load_weights('weights/wdsr-32-x4/weights.h5')

    train_ds_small_batch = catesr_train.dataset(batch_size=4, random_transform=True, shuffle_buffer_size=500)

    # Fine-tune EDSR model via SRGAN training.
    gan_trainer = SrganTrainer(generator=generator, discriminator=discriminator())
    gan_trainer.train(train_ds_small_batch, steps=100000)

    new_weights_file = os.path.join(weights_dir, 'weights_fine_tuned.h5')
    generator.save_weights(new_weights_file)
import os
from data import CATESR

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from model.wdsr import wdsr_b


if __name__ == "__main__":

    depth = 32    # Number of residual blocks
    scale = 4    # Super-resolution factor

    # Location of model weights (needed for demo)
    weights_dir = f'weights/wdsr-{depth}-x{scale}'
    weights_file = os.path.join(weights_dir, 'weights.h5')

    os.makedirs(weights_dir, exist_ok=True)


    catesr_train = CATESR(subset='train', images_dir='/Users/cate/data/gans/images_rgb',
                          caches_dir='/Users/cate/data/gans/caches_rgb')
    catesr_valid = CATESR(subset='valid', images_dir='/Users/cate/data/gans/images_rgb',
                          caches_dir='/Users/cate/data/gans/caches_rgb')


    # catesr_train = CATESR(subset='train', images_dir='/home/ec2-user/gan/data/images_rgb',
    #                       caches_dir='/home/ec2-user/gan/data/caches_rgb')
    # catesr_valid = CATESR(subset='valid', images_dir='/home/ec2-user/gan/data/images_rgb',
    #                       caches_dir='/home/ec2-user/gan/data/caches_rgb')

    train_ds = catesr_train.dataset(batch_size=16, random_transform=True, shuffle_buffer_size=500)
    valid_ds = catesr_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

    # Custom WDSR B model (0.62M parameters)
    model_wdsr = wdsr_b(scale=4, num_res_blocks=32)

    # Adam optimizer with a scheduler that halfs learning rate after 200,000 steps
    optim_wdsr = Adam(learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4]))

    # Compile and train model for 300,000 steps with L1 pixel loss
    model_wdsr.compile(optimizer=optim_wdsr, loss='mean_absolute_error')
    model_wdsr.fit(train_ds, epochs=300, steps_per_epoch=1000)

    # Save weights
    model_wdsr.save_weights(os.path.join(weights_dir, 'weights-wdsr-b-32-x4.h5'))
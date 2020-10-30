# Spectral normalization for generative adversarial networks
# https://arxiv.org/pdf/1802.05957
# The relativistic discriminator: a key element missing from standard GAN
# https://arxiv.org/pdf/1807.00734


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import datetime
import pathlib
import sys

# tf.debugging.set_log_device_placement(True)
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

from CustomUtils import *

BATCH_SIZE = 64
LR = 1e-4
N_CRITIC = 1
EPOCHS = 20

path = "./datasets/img_align_celeba"
files = list(pathlib.Path(path).glob('*'))


def preprocess_img(filename):
    img = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, [64, 64])
    # project pixel value into [-1, 1]
    img = (tf.cast(img, 'float32') - 127.5) / 127.5
    return img


train_dataset = tf.data.Dataset.from_tensor_slices([str(f) for f in files])
train_dataset = train_dataset.shuffle(210000)
train_dataset = train_dataset.map(preprocess_img)
train_dataset = train_dataset.shuffle(8192, reshuffle_each_iteration=True).batch(BATCH_SIZE)


def get_generate_model():
    input = tf.keras.Input(shape=(256,), dtype=tf.float32)

    x = layers.Dense(units=4 * 4 * 512, activation=None, use_bias=False, kernel_initializer='glorot_normal')(input)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.Activation('relu')(x)
    x = layers.Reshape((4, 4, 512))(x)

    x = layers.Conv2DTranspose(filters=256,
                               kernel_size=(5, 5),
                               strides=(2, 2),
                               padding='same',
                               output_padding=(1, 1),
                               use_bias=False,
                               kernel_initializer='glorot_normal')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    # output (8,8,256)

    x = layers.Conv2DTranspose(filters=128,
                               kernel_size=(5, 5),
                               strides=(2, 2),
                               padding='same',
                               output_padding=(1, 1),
                               use_bias=False,
                               kernel_initializer='glorot_normal')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    # output (16,16,128)

    x = layers.Conv2DTranspose(filters=64,
                               kernel_size=(5, 5),
                               strides=(2, 2),
                               padding='same',
                               output_padding=(1, 1),
                               use_bias=False,
                               kernel_initializer='glorot_normal')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    # output (32,32,64)

    x = layers.Conv2DTranspose(filters=32,
                               kernel_size=(5, 5),
                               strides=(2, 2),
                               padding='same',
                               output_padding=(1, 1),
                               use_bias=True,
                               kernel_initializer='glorot_normal')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    # output (64,64,32)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=3,
                      kernel_size=(5, 5),
                      strides=(1, 1),
                      padding='same',
                      use_bias=True,
                      kernel_initializer='glorot_normal')(x)
    x = layers.Activation('tanh')(x)
    # output (64,64,3)

    model = tf.keras.models.Model(inputs=input, outputs=x)
    return model


g = get_generate_model()


def get_discrimnate_model():
    input = tf.keras.Input(shape=(64, 64, 3), dtype=tf.float32)
    x = ConvSN2D(filters=64,
                      kernel_size=(5, 5),
                      strides=(2, 2),
                      padding='same',
                      use_bias=True,
                      kernel_initializer='glorot_normal')(input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # output (32, 32, 64)

    x = ConvSN2D(filters=128,
                      kernel_size=(5, 5),
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer='glorot_normal')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # x = layers.Dropout(0.3)(x)
    # output (16, 16, 128)

    x = ConvSN2D(filters=256,
                      kernel_size=(5, 5),
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer='glorot_normal')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # x = layers.Dropout(0.3)(x)
    # output (8, 8, 256)

    x = ConvSN2D(filters=512,
                      kernel_size=(5, 5),
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer='glorot_normal')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # x = layers.Dropout(0.3)(x)
    # output(4, 4, 512)

    x = layers.Flatten()(x)
    x = DenseSN(units=1, activation=None, use_bias=True, kernel_initializer='glorot_normal')(x)

    model = tf.keras.models.Model(inputs=input, outputs=x)
    return model


d = get_discrimnate_model()


def _f1(x):
    return tf.square(x - 1.)


def _f2(x):
    return tf.square(x + 1.)


def discriminator_loss(realX, fakeX):
    cr_bar = tf.reduce_mean(d(realX), keepdims=True)
    cf_bar = tf.reduce_mean(d(fakeX), keepdims=True)
    return tf.reduce_mean(
        _f1(d(realX) - cf_bar) + _f2(d(fakeX) - cr_bar)
    )


def generator_loss(realX, fakeX):
    cr_bar = tf.reduce_mean(d(realX), keepdims=True)
    cf_bar = tf.reduce_mean(d(fakeX), keepdims=True)
    return tf.reduce_mean(
        _f1(d(fakeX) - cr_bar) + _f2(d(realX) - cf_bar)
    )


g_optimizer = tf.keras.optimizers.Adam(LR, beta_1=0.5, beta_2=0.999)
d_optimizer = tf.keras.optimizers.Adam(LR, beta_1=0.5, beta_2=0.999)

checkpoint_dir = './faceGAN_ckpt'
ckpt = tf.train.Checkpoint(g_optimizer=g_optimizer,
                           d_optimizer=d_optimizer,
                           g=g, d=d)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(
    log_dir + "faceGAN/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))


@tf.function
def g_train_step(X):
    noise = tf.random.normal([X.shape[0], 256])
    with tf.GradientTape() as gen_tape:
        fakeX = g(noise, training=True)
        g_loss = generator_loss(X, fakeX)
    gradient_g = gen_tape.gradient(g_loss, g.trainable_variables)
    g_optimizer.apply_gradients(zip(gradient_g, g.trainable_variables))
    return g_loss


# note that the first dim of X is not necessarily BATCH_SIZE
@tf.function
def d_train_step(X):
    noise = tf.random.normal([X.shape[0], 256])
    with tf.GradientTape() as disc_tape:
        fakeX = g(noise, training=True)
        d_loss = discriminator_loss(X, fakeX)
    gradient_d = disc_tape.gradient(d_loss, d.trainable_variables)
    d_optimizer.apply_gradients(zip(gradient_d, d.trainable_variables))
    return d_loss


def write_logs(g_loss, d_loss, n_steps):
    with summary_writer.as_default():
        tf.summary.scalar('gen_loss', g_loss, step=n_steps)
        tf.summary.scalar('disc_loss', d_loss, step=n_steps)
        tf.summary.scalar('total_loss', g_loss + d_loss, step=n_steps)


n_steps = 0


def train(train_dataset, epochs):
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    global n_steps
    bar = ProgressBar(n_steps)
    for epoch in range(epochs):
        print("epoch: {}".format(epoch + 1))
        starttime = datetime.datetime.now().timestamp()
        it = iter(train_dataset)
        has_next = True
        while has_next:
            for _ in range(N_CRITIC):
                try:
                    x = next(it)
                    d_loss = d_train_step(x)
                except StopIteration:
                    has_next = False
                    d_loss = 0.
                    break

            try:
                x = next(it)
                g_loss = g_train_step(x)
            except StopIteration:
                has_next = False
                g_loss = 0.
            n_steps += 1
            bar.step_forward()
            if n_steps % 100 == 0:
                write_logs(g_loss, d_loss, n_steps // 100)
        print('\n', end='')
        endtime = datetime.datetime.now().timestamp()
        print("epoch {} end. take {} seconds.".format(epoch + 1, endtime - starttime))

        ckpt_manager.save()

        show_sample_image(epoch + 1)


test_noise = tf.random.normal([25, 256])


def show_sample_image(n_epoch):
    global test_noise
    fakeX = g(test_noise, training=False)
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    for i in range(fakeX.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow((fakeX[i, :, :, :].numpy() * 127.5 + 127.5).astype('uint8'))
        plt.axis('off')
    plt.savefig(checkpoint_dir + '/epoch_{}.png'.format(n_epoch))
    plt.show()


class ProgressBar:
    def __init__(self, steps):
        self.steps = steps
        self.toolbar_width = 100

    def printBar(self):
        c = self.steps % 1000 * self.toolbar_width // 1000
        sys.stdout.write('\r')
        sys.stdout.write('[%s%s]' % ('>' * c, '.' * (self.toolbar_width - c)))
        sys.stdout.write(str(self.steps))
        sys.stdout.flush()

    def step_forward(self):
        self.steps += 1
        self.printBar()


train(train_dataset, EPOCHS)

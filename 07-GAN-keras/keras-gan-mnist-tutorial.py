import argparse

import numpy as np
# import matplotlib.pyplot as plt

from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K


K.set_image_data_format('channels_first')


class Data:
    """
    Define dataset for training GAN
    """
    def __init__(self, batch_size, z_input_dim):
        # load mnist dataset
        # 이미지는 보통 -1~1 사이의 값으로 normalization : generator의 outputlayer를 tanh로
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        self.x_data = ((X_train.astype(np.float32) - 127.5) / 127.5)
        self.x_data = self.x_data.reshape((self.x_data.shape[0], 1) + self.x_data.shape[1:])
        self.batch_size = batch_size
        self.z_input_dim = z_input_dim

    def get_real_sample(self):
        """
        get real sample mnist images

        :return: batch_size number of mnist image data
        """
        return self.x_data[np.random.randint(0, self.x_data.shape[0], size=self.batch_size)]

    def get_z_sample(self, sample_size):
        """
        get z sample data

        :return: random z data (batch_size, z_input_dim) size
        """
        return np.random.uniform(-1.0, 1.0, (sample_size, self.z_input_dim))


class GAN:
    def __init__(self, learning_rate, z_input_dim):
        self.learning_rate = learning_rate
        self.z_input_dim = z_input_dim
        self.D = self.discriminator()
        self.G = self.generator()
        self.GD = self.combined()

    def discriminator(self):
        D = Sequential()
        D.add(Conv2D(256, (5, 5),
                     padding='same',
                     input_shape=(1, 28, 28),
                     kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        D.add(LeakyReLU(0.2))
        D.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        D.add(Dropout(0.3))
        D.add(Conv2D(512, (5, 5), padding='same'))
        D.add(LeakyReLU(0.2))
        D.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        D.add(Dropout(0.3))
        D.add(Flatten())
        D.add(Dense(256))
        D.add(LeakyReLU(0.2))
        D.add(Dropout(0.3))
        D.add(Dense(1, activation='sigmoid'))

        adam = Adam(lr=self.learning_rate, beta_1=0.5)
        D.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        # discriminator = Sequential()
        # discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        # discriminator.add(LeakyReLU(0.2))
        # discriminator.add(Dropout(0.3))
        # discriminator.add(Dense(512))
        # discriminator.add(LeakyReLU(0.2))
        # discriminator.add(Dropout(0.3))
        # discriminator.add(Dense(256))
        # discriminator.add(LeakyReLU(0.2))
        # discriminator.add(Dropout(0.3))
        # discriminator.add(Dense(1, activation='sigmoid'))
        # discriminator.compile(loss='binary_crossentropy', optimizer=adam)
        #
        # d_input = Input(shape=shp)
        # H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(d_input)
        # H = LeakyReLU(0.2)(H)
        # H = Dropout(dropout_rate)(H)
        # H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(H)
        # H = LeakyReLU(0.2)(H)
        # H = Dropout(dropout_rate)(H)
        # H = Flatten()(H)
        # H = Dense(256)(H)
        # H = LeakyReLU(0.2)(H)
        # H = Dropout(dropout_rate)(H)
        # d_V = Dense(2, activation='softmax')(H)
        # discriminator = Model(d_input, d_V)
        # discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)

        return D

    def generator(self):
        # G = Sequential()
        # G.add(Dense(512, input_dim=self.z_input_dim))
        # G.add(LeakyReLU(0.2))
        # G.add(Dense(128 * 7 * 7))
        # G.add(LeakyReLU(0.2))
        # G.add(BatchNormalization())
        # G.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        # G.add(UpSampling2D(size=(2, 2)))
        # G.add(Conv2D(64, (5, 5), padding='same', activation='tanh'))
        # G.add(UpSampling2D(size=(2, 2)))
        # G.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))

        G = Sequential()
        G.add(Dense(512, input_dim=self.z_input_dim))
        G.add(LeakyReLU(0.2))
        G.add(Dense(128 * 7 * 7))
        G.add(LeakyReLU(0.2))
        G.add(BatchNormalization())
        G.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        G.add(UpSampling2D(size=(2, 2)))
        G.add(Conv2D(64, (5, 5), padding='same', activation='tanh'))
        G.add(UpSampling2D(size=(2, 2)))
        G.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))

        adam = Adam(lr=self.learning_rate, beta_1=0.5)
        G.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        # model = models.Sequential()
        # model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        # model.add(layers.Dense(128 * 7 * 7, activation='tanh'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        # model.add(layers.UpSampling2D(size=(2, 2)))
        # model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
        # model.add(layers.UpSampling2D(size=(2, 2)))
        # model.add(layers.Conv2D(1, (5, 5), padding='same', activation='tanh'))
        #
        #
        # generator = Sequential()
        # generator.add(Dense(256, input_dim=self.z_input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        # generator.add(LeakyReLU(0.2))
        # generator.add(Dense(512))
        # generator.add(LeakyReLU(0.2))
        # generator.add(Dense(1024))
        # generator.add(LeakyReLU(0.2))
        # generator.add(Dense(784, activation='tanh'))
        # generator.compile(loss='binary_crossentropy', optimizer=adam)
        #
        # nch = 200
        # g_input = Input(shape=[100])
        # H = Dense(nch * 14 * 14, init='glorot_normal')(g_input)
        # H = BatchNormalization(mode=2)(H)
        # H = Activation('relu')(H)
        # H = Reshape([nch, 14, 14])(H)
        # H = UpSampling2D(size=(2, 2))(H)
        # H = Convolution2D(nch / 2, 3, 3, border_mode='same', init='glorot_uniform')(H)
        # H = BatchNormalization(mode=2)(H)
        # H = Activation('relu')(H)
        # H = Convolution2D(nch / 4, 3, 3, border_mode='same', init='glorot_uniform')(H)
        # H = BatchNormalization(mode=2)(H)
        # H = Activation('relu')(H)
        # H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
        # g_V = Activation('sigmoid')(H)
        # generator = Model(g_input, g_V)
        # generator.compile(loss='binary_crossentropy', optimizer=opt)
        # generator.summary()

        return G

    def combined(self):
        G, D = self.G, self.D
        D.trainable = False
        GD = Sequential()
        GD.add(G)
        GD.add(D)

        adam = Adam(lr=self.learning_rate, beta_1=0.5)
        GD.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        D.trainable = True

        # discriminator.trainable = False
        # ganInput = Input(shape=(randomDim,))
        # x = generator(ganInput)
        # ganOutput = discriminator(x)
        # gan = Model(inputs=ganInput, outputs=ganOutput)
        # gan.compile(loss='binary_crossentropy', optimizer=adam)

        return GD


class Model:
    def __init__(self, args):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.z_input_dim = args.z_input_dim
        self.data = Data(self.batch_size, self.z_input_dim)

        # the reason why D, G differ in iter : Generator needs more training than Discriminator
        self.n_iter_D = args.n_iter_D
        self.n_iter_G = args.n_iter_G
        self.gan = GAN(self.learning_rate, self.z_input_dim)
        self.d_loss = []
        self.g_loss = []

        # print status
        batch_count = self.data.x_data.shape[0] / self.batch_size
        print('Epochs:', self.epochs)
        print('Batch size:', self.batch_size)
        print('Batches per epoch:', batch_count)
        print('Learning rate:', self.learning_rate)
        print('Image data format:', K.image_data_format())

    def fit(self):
        for epoch in range(self.epochs):

            # train discriminator by real data
            dloss = 0
            for iter in range(self.n_iter_D):
                dloss = self.train_D()

            # train GD by generated fake data
            gloss = 0
            for iter in range(self.n_iter_G):
                gloss = self.train_G()

            # save loss data
            print('Discriminator loss:', str(dloss))
            print('Generator loss:', str(gloss))

    def train_D(self):
        """
        train Discriminator
        """

        # Real data
        real = self.data.get_real_sample()

        # Generated data
        z = self.data.get_z_sample(self.batch_size)
        generated_images = self.gan.G.predict(z)

        # labeling and concat generated, real images
        x = np.concatenate((real, generated_images), axis=0)
        y = [0.9] * self.batch_size + [0] * self.batch_size

        # train discriminator
        self.gan.D.trainable = True
        loss = self.gan.D.train_on_batch(x, y)
        return loss

    def train_G(self):
        """
        train Generator
        """

        # Generated data
        z = self.data.get_z_sample(self.batch_size)

        # labeling
        y = [1] * self.batch_size

        # train generator
        self.gan.D.trainable = False
        loss = self.gan.GD.train_on_batch(z, y)
        return loss

def main():
    # set hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for networks')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Epochs for the networks')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--z_input_dim', type=int, default=100,
                        help='Input dimension for the generator.')
    parser.add_argument('--n_iter_D', type=int, default=1,
                        help='training iteration for D')
    parser.add_argument('--n_iter_G', type=int, default=1,
                        help='training iteration for G')
    args = parser.parse_args()

    # run model
    model = Model(args)
    model.fit()


if __name__ == '__main__':
    main()
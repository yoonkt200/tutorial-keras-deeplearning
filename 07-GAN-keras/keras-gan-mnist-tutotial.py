import argparse

import numpy as np
# import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K


class GAN:
    def __init__(self, learning_rate, input_dim):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.D = self.discriminator()
        self.G = self.generator()
        self.GD = self.combined()

    def discriminator(self):
        pass

    def generator(self):
        pass

    def combined(self):
        pass


class Model:
    def __init__(self, args):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        # the reason why D, G differ in iter : Generator needs more training than Discriminator
        self.n_iter_D = args.n_iter_D
        self.n_iter_G = args.n_iter_G
        self.gan = GAN(self.learning_rate, args.input_dim)

    def fit(self):
        pass

    def plot_loss_graph(self, g_loss, d_loss):
        """
        Save training loss graph

        :param g_loss:
        :param d_loss:
        :return:
        """
        plt.figure(figsize=(10, 8))
        plt.plot(d_loss, label='Discriminator loss')
        plt.plot(g_loss, label='Generator loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('images/mnist_gan_loss_graph.png')

    def plt_generate_images(self, epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
        """
        Save generated mnist images

        :param epoch:
        :param generator:
        :param examples:
        :param dim:
        :param figsize:
        :return:
        """
        noise = np.random.normal(0, 1, size=[examples, randomDim])
        generatedImages = generator.predict(noise)
        generatedImages = generatedImages.reshape(examples, 28, 28)

        plt.figure(figsize=figsize)
        for i in range(generatedImages.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('images/mnist_generated_image_epoch_%d.png' % epoch)

    def save_gan_model(self, generator, discriminator, epoch):
        """
        Save model trained generator, discriminator

        :param generator:
        :param discriminator:
        :param epoch:
        :return:
        """
        generator.save('mnist_models/mnist_generator_epoch_%d.h5' % epoch)
        discriminator.save('mnist_models/mnist_discriminator_epoch_%d.h5' % epoch)


def main():
    # set hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mu', type=float, default=4,
                        help='Mean of population data')
    parser.add_argument('--sigma', type=float, default=1.25,
                        help='Sigma of population data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for networks')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Epochs for the networks')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--input_dim', type=int, default=100,
                        help='Input dimension for the generator.')
    parser.add_argument('--n_iter_D', type=int, default=1,
                        help='training iteration for D')
    parser.add_argument('--n_iter_G', type=int, default=5,
                        help='training iteration for G')
    args = parser.parse_args()

    # run model
    model = Model(args)
    model.fit()


if __name__ == '__main__':
    main()
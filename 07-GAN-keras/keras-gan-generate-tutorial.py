import argparse

import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras.layers import Dense, Conv1D, Reshape, Flatten, Lambda
from keras.optimizers import Adam
from keras import backend as K


class Data:
    """
    Define dataset for training GAN
    """
    def __init__(self, mu, sigma, input_dim):
        """
        ex) real_sample(n) -> (n, input_dim) data generate
        :param mu: mean of data
        :param sigma: sigma of data
        :param input_dim: input dimension
        """
        self.real_sample = lambda n_batch: np.random.normal(mu, sigma, (n_batch, input_dim))
        self.in_sample = lambda n_batch: np.random.rand(n_batch, input_dim)


class GAN:
    """
    Define GAN model
    """
    def __init__(self):
        pass


class Model:
    """
    Learner class
    """
    def __init__(self, args):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.data = Data(args.mu, args.sigma, args.input_dim)
        # the reason why D, G differ in iter : Generator needs more training than Discriminator
        self.n_iter_D = args.n_iter_D
        self.n_iter_G = args.n_iter_G
        self.model = GAN()

    def fit(self):
        for epoch in range(self.epochs):
            pass
        pass


def main():
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
    model = Model(args)
    model.fit()


if __name__ == '__main__':
    main()
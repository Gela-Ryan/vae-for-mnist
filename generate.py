# -*- coding: UTF-8 -*-

import os
import sys

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from train_vae_on_mnist import get_vae_model
from vae_model import VariationalAutoEncoder

def generate_mnist_samples():
    '''
        Generate mnist samples from a grid of VAE
        latent space. You should be able to observe
        the semantic meaning of each dimension.
    '''

    ## Get the decoder of the trained vae
    vae_model = get_vae_model()
    weight_path = os.path.join('models', 'weights.model')
    vae_model.load_weights(weight_path)
    decoder = vae_model.build_decoder()

    ## Generate samples from the latent space
    num_per_row, num_per_col = 20, 20
    batch_size = 16
    digit_size = 28
    figure = np.zeros((digit_size * num_per_row, digit_size * num_per_col))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, num_per_row))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, num_per_col))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = decoder.predict(z_sample, batch_size=batch_size)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(figure, cmap='Greys_r')
    save_root = 'generations'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    plt.savefig(os.path.join(save_root, 'sample.png'), dpi=300)
    # dpi 指定分辨率

if __name__ == '__main__':
    generate_mnist_samples()

# -*- coding: UTF-8 -*-

import os
import sys

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from train_vae_on_mnist import get_vae_model
from tensorflow.keras.datasets import mnist
from vae_model import VariationalAutoEncoder

def get_mnist_data():
    '''
        Get the mnist dataset from
        tensorflow.keras built-in
    '''
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))
    return x_test, y_test

def generate_latent_z():
    vae_model = get_vae_model()
    weight_path = os.path.join('models', 'weights.model')
    vae_model.load_weights(weight_path)
    encoder = vae_model.build_encoder()

    x_test, y_test = get_mnist_data()
    # print(y_test)
    color_dict = {1: 'r', 2: 'k', 3: 'grey', 4: 'b', 5: 'g', 6: 'y', 7: 'c', 8: 'm', 9: 'orange', 0: 'pink'}

    z_sample = encoder.predict(x_test)
    print(z_sample.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for index in range(z_sample.shape[0]):
        # print(z_sample[index,0], z_sample[index,1], color_dict[y_test[index]])
        ax.scatter(z_sample[index, 0], z_sample[index, 1], z_sample[index, 2],c=color_dict[y_test[index]])
    # plt.scatter(z_sample[:, 0], z_sample[:, 1], color=color_dict[y_test], cmap=plt.cm.Paired)
    ax.set_xlabel('1d')
    ax.set_ylabel('2d')
    ax.set_zlabel('3d')
    save_root = "generations"
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    plt.savefig(os.path.join(save_root, 'latent_z.png'))
    plt.show()


if __name__ == '__main__':
    # x_test, y_test = get_mnist_data()
    # print(x_test.shape)
    generate_latent_z()

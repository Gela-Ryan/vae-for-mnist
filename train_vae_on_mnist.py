# -*- coding: UTF-8 -*-

import os
import sys

from tensorflow.keras.datasets import mnist
from vae_model import VariationalAutoEncoder

def get_vae_model():
    '''
        Build the VAE model for mnist.
        Latent dimension: 2
    '''
    vae = VariationalAutoEncoder(
        img_shape=(28, 28, 1),
        enc_conv_sizes=[32, 64, 64, 64],
        enc_hidden_sizes=[32],
        kernel_size=3,
        activation='relu',
        latent_dim=3
    )
    return vae

def get_mnist_data():
    '''
        Get the mnist dataset from tensorflow.keras built-in
    '''
    (x_train, _), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    # 将图片数值用float32存储减伤内存budget， / 255.是将取值区间(0, 255)归一化成(0, 1)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))
    return x_train, x_test

def train_vae():
    '''
        train VAE on mnist dataset and save weight
    '''
    ## prepare VAE
    vae = get_vae_model()
    vae_train = vae.build_model()
    vae_train.compile(optimizer='rmsprop', loss=None)

    ## prepare dataset
    x_train, x_test = get_mnist_data()

    ## Train the VAE model with rmsprop
    vae_train.fit(
        x=x_train, y=None,
        shuffle=True,
        epochs=10,
        batch_size=16,
        validation_data=(x_test, None))

    ## Save weight
    save_root = 'models'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    vae.save_weights(os.path.join(save_root, 'weights.model'))

if __name__ == '__main__':
    train_vae()

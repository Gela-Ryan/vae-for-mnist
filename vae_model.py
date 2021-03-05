# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from tensorflow.python.keras.engine import network
from tensorflow_probability import distributions as tfp

class AddGaussianLoss(layers.Layer):
    '''
        Add the KL divergence between variational distribution and the prior to loss.
    '''
    def __init__(self, **kwargs):
        ## **kwarg 代表接受任意数量指定key参数（e.g. __init__(name=apple, color=red)）存在字典中
        ## *arg 代表接受任意数量未指定key参数(e.g. test(apple, red))存在列表中
        super(AddGaussianLoss, self).__init__(**kwargs)
        ## 找到 AddGaussianLoss 的父类 layers.Layer， 然后将类 AddGaussianLoss 的对象转化为类 layers.Layer 的对象
        self.lamb_kl = self.add_weight(shape=(),
                                       name='lamb_kl',
                                       initializer='ones',
                                       trainable=False)
        # add_weight: creates layer state variables that do not depend on input shapes

    def call(self, inputs):
        '''
            Defines the computation from inputs to output
            mu & std must be vectors, e.g. inputs = [[0.], [1.]]
        '''
        mu, std = inputs
        var_dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        pri_dist = tfp.MultivariateNormalDiag(loc=K.zeros_like(mu),
                                              scale_diag=K.ones_like(std))
        # zeros_like, ones_like: 返回一个类似x的全0，1的张量

        kl_loss = self.lamb_kl * K.mean(tfp.kl_divergence(var_dist, pri_dist))
        return kl_loss

class ReparameterizeGaussian(layers.Layer):
    '''
        Rearameterization trick for Gaussian
        mu & std must be vectors, e.g. states = [[0.], [1.]]
    '''

    def __init__(self, **kwargs):
        super(ReparameterizeGaussian, self).__init__(**kwargs)

    def call(self, states):
        mu, std = states
        dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        # 参数 scale_diag 为协方差矩阵的对角，即各个维度自身的方差
        return dist.sample()

class SamplerGaussian(network.Network):
    '''
        Sample from the variational Gaussian, and add its KL with the prior to loss
    '''
    def __init__(self, **kwargs):
        super(SamplerGaussian, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        # 字典函数get获得键"name"的值，没有则返回None
        self.rep_gauss = ReparameterizeGaussian()

    def build(self, input_shapes):
        mean = layers.Input(input_shapes[0][1:])
        # 定义输入层，输入的尺寸
        std = layers.Input(input_shapes[1][1:])
        sample = self.rep_gauss([mean, std])
        self._init_graph_network([mean, std], [sample], name=self.m_name)
        super(SamplerGaussian, self).build(input_shapes)

class VariationalAutoEncoder():
    '''
        VAE for mnist image generation
    '''
    def __init__(self,
                 img_shape,
                 enc_conv_sizes,
                 enc_hidden_sizes,
                 kernel_size,
                 activation,
                 latent_dim):
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        ################## layers for encoder
        ## conv layer ##
        with tf.name_scope('enc_conv_layers'):
            self.enc_convs = []
            for i, conv_size in enumerate(enc_conv_sizes):
                self.enc_convs.append(layers.Conv2D(conv_size, kernel_size, activation=activation, padding='same', name='enc_conv_{}'.format(i+1)) if i != 1
                                      else layers.Conv2D(conv_size, kernel_size, activation=activation, strides =(2,2), padding='same', name="enc_conv_{}".format(i+1)))
        ## dense layer ##
        with tf.name_scope('enc_dense_layers'):
            self.enc_denses = []
            for i, hidden_size in enumerate(enc_hidden_sizes):
                self.enc_denses.append(layers.Dense(hidden_size, activation=activation, name="enc_dense_{}".format(i+1)))

        self.latent_mean = layers.Dense(latent_dim, name='latent_mean')
        self.latent_logstd = layers.Dense(latent_dim, name='latent_logstd')
        ##################

        ################## layers for reparameterization trick
        with tf.name_scope('repara_layers'):
            self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2), name='clip')
            # clip将x中超出区间[-20,2]的元素强制裁剪为边界
            self.exp = layers.Lambda(lambda x:K.exp(x), name='exp')
            # 元素级exp运算
            self.sampler_gauss = SamplerGaussian()
        ##################

        ################## layers for decoder
        with tf.name_scope('dec_dense_layer'):
            last_conv_shape = (int(img_shape[0]/2), int(img_shape[1]/2), enc_conv_sizes[-1])
            hidden_size = np.prod(last_conv_shape)
            self.dec_dense = layers.Dense(hidden_size, activation=activation, name='dec_dense')
            self.reshape = layers.Reshape(last_conv_shape)
        with tf.name_scope('dec_deconv_layers'):
            self.dec_convs = [] # 反卷积
            self.dec_convs.append(layers.Conv2DTranspose(32, kernel_size, activation=activation, padding='same', strides=(2,2), name='dec_conv_1'))
            self.dec_convs.append(layers.Conv2D(img_shape[-1], kernel_size, activation='sigmoid', padding = 'same', name='dec_conv_2'))

    def build_model(self):
        '''
            Build the VAE model for training, includes
            reconstruction loss and KL loss with prior
        '''
        if hasattr(self, 'vae_model'):
            return self.vae_model
        input_img = layers.Input(shape=self.img_shape, name='input_img')
        ################## input into encoder:conv
        x = self.enc_convs[0](input_img)
        for enc_conv in self.enc_convs[1:]:
            x = enc_conv(x)
        ##################

        x = layers.Flatten(name='flatten_layer')(x)

        ################## input into encoder:dense
        for enc_dense in self.enc_denses:
            x = enc_dense(x)
        ##################

        z_mean = self.latent_mean(x)
        z_std = self.exp(self.clip(self.latent_logstd(x)))
        z = self.sampler_gauss([z_mean, z_std])
        # print(z.shape)

        ################## input into decoder:dense
        x = self.reshape(self.dec_dense(z))
        ##################

        ##################  input into decoder:conv
        for dec_conv in self.dec_convs:
            x = dec_conv(x)
        ##################

        self.vae_model = models.Model(input_img, x) # 根据参数input(input_img) and output(x)建立模型
        self.vae_model.add_loss(2.5e-5 * AddGaussianLoss()([z_mean, z_std]))
        # AddGaussianLoss 自定义的关于z分布的评估函数
        self.vae_model.add_loss(K.mean(metrics.binary_crossentropy(K.flatten(input_img), K.flatten(x))))
        # K.mean 取均值， metrics.binary_conssentropy 评估函数， flatten展成一个1D张量

        return self.vae_model

    def build_encoder(self, return_sample=True):
        '''
            Build the encoder that encodes an mnist
            image into the hidden representation
        '''
        if return_sample and hasattr(self, "encoder_sample"):
            return self.encoder_sample
        if not return_sample and hasattr(self, "encoder_params"):
            return self.encoder_params
        ################## input into encoder:conv
        input_img = layers.Input(shape=self.img_shape)
        x = self.enc_convs[0](input_img)
        for enc_conv in self.enc_convs[1:]:
            x = enc_conv(x)
        ##################

        x = layers.Flatten()(x)

        ################## input into encoder:dense
        for enc_dense in self.enc_denses:
            x = enc_dense(x)
        ##################

        z_mean = self.latent_mean(x)
        z_std = self.exp(self.clip(self.latent_logstd(x)))
        z = self.sampler_gauss([z_mean, z_std])

        if return_sample:
            self.encoder_sample = models.Model(input_img, z)
            return self.encoder_sample
        else:
            self.encoder_params = model.Model(input_img, [z_mean, z_std])
            return self.encoder_params

    def build_decoder(self):
        '''
            Build the decoder that decodes an hidden
            variable into an mnist image
        '''
        if hasattr(self, "decoder"):
            return self.decoder
        input_z = layers.Input(shape=(self.latent_dim,))
        x = self.reshape(self.dec_dense(input_z))
        for dec_conv in self.dec_convs:
            x = dec_conv(x)
        self.decoder = models.Model(input_z, x)
        return self.decoder

    def save_weights(self, weight_path):
        self.build_model().save_weights(weight_path)

    def load_weights(self, weight_path):
        self.build_model().load_weights(weight_path)


if __name__ == '__main__':
    with tf.Session() as sess:
        vae = VariationalAutoEncoder(
            img_shape=(28, 28, 1),
            enc_conv_sizes=[32, 64, 64, 64],
            enc_hidden_sizes=[32],
            kernel_size=3,
            activation="relu",
            latent_dim=2
        )
        vae_train = vae.build_model()
        merged = tf.summary.merge_all()
        vae_train.compile(optimizer='rmsprop', loss=None)
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(tf.global_variables_initializer())

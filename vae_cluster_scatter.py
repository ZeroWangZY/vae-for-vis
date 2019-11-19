#! -*- coding: utf-8 -*-

import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
import imageio, os
import json

batch_size = 100
latent_dim = 20
epochs = 300
num_classes = 9
img_dim = 64
filters = 16
intermediate_dim = 256
kernel_size = 5

# 加载数据集
scatters_data = []
with open("data/scatters_" + str(img_dim) + ".json", 'r') as load_f:
    scatters_data = json.load(load_f)
x_train = np.array(scatters_data)
# mean = np.mean(x_train)
# std = np.std(x_train)
# x_train = (x_train - mean) / (std)
x_train = x_train.reshape((-1, img_dim, img_dim, 1))

# 搭建模型
x = Input(shape=(img_dim, img_dim, 1))
h = x

for i in range(2):
    filters *= 2
    h = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               padding='same')(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same')(h)
    h = LeakyReLU(0.2)(h)

h_shape = K.int_shape(h)[1:]
h = Flatten()(h)
z_mean = Dense(latent_dim)(h)  # p(z|x)的均值
z_log_var = Dense(latent_dim)(h)  # p(z|x)的方差

encoder = Model(x, z_mean)  # 通常认为z_mean就是所需的隐变量编码

z = Input(shape=(latent_dim, ))
h = z
h = Dense(np.prod(h_shape))(h)
h = Reshape(h_shape)(h)

for i in range(2):
    h = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same')(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        padding='same')(h)
    h = LeakyReLU(0.2)(h)
    filters //= 2

x_recon = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same')(h)

decoder = Model(z, x_recon)  # 解码器
generator = decoder

z = Input(shape=(latent_dim, ))
y = Dense(intermediate_dim, activation='relu')(z)
y = Dense(num_classes, activation='softmax')(y)

classfier = Model(z, y)  # 隐变量分类器


# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon


# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_log_var])
x_recon = decoder(z)
y = classfier(z)


class Gaussian(Layer):
    """这是个简单的层，定义q(z|y)中的均值参数，每个类别配一个均值。
    然后输出“z - 均值”，为后面计算loss准备。
    """
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)

    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_classes, latent_dim),
                                    initializer='zeros')

    def call(self, inputs):
        z = inputs  # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z - K.expand_dims(self.mean, 0)

    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])


gaussian = Gaussian(num_classes)
z_prior_mean = gaussian(z)

# 建立模型
vae = Model(x, [x_recon, z_prior_mean, y])

# 下面一大通都是为了定义loss
z_mean = K.expand_dims(z_mean, 1)
z_log_var = K.expand_dims(z_log_var, 1)

lamb = 2.5  # 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
xent_loss = 0.5 * K.mean((x - x_recon)**2, 0)
kl_loss = -0.5 * (z_log_var - K.square(z_prior_mean))
kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)
cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
vae_loss = lamb * K.sum(xent_loss) + K.sum(kl_loss) + K.sum(cat_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size)

means = K.eval(gaussian.mean)
x_train_encoded = encoder.predict(x_train)
y_train_pred = classfier.predict(x_train_encoded).argmax(axis=1)
print(y_train_pred)


def cluster_sample(path, category=0):
    """观察被模型聚为同一类的样本
    """
    n = 5
    figure = np.zeros((img_dim * n, img_dim * n))
    idxs = np.where(y_train_pred == category)[0]
    if len(idxs) == 0:
        return
    for i in range(n):
        for j in range(n):
            digit = x_train[np.random.choice(idxs)]
            digit = digit.reshape((img_dim, img_dim))
            figure[i * img_dim:(i + 1) * img_dim, j * img_dim:(j + 1) *
                   img_dim] = digit
    imageio.imwrite(path, figure * 255)


def random_sample(path, category=0, std=1):
    """按照聚类结果进行条件随机生成
    """
    n = 10
    figure = np.zeros((img_dim * n, img_dim * n))
    for i in range(n):
        for j in range(n):
            noise_shape = (1, latent_dim)
            z_sample = np.array(
                np.random.randn(*noise_shape)) * std + means[category]
            x_recon = generator.predict(z_sample)
            digit = x_recon[0].reshape((img_dim, img_dim))
            figure[i * img_dim:(i + 1) * img_dim, j * img_dim:(j + 1) *
                   img_dim] = digit
    imageio.imwrite(path, figure * 255)


if not os.path.exists('samples'):
    os.mkdir('samples')

for i in range(num_classes):
    cluster_sample(u'samples/聚类类别_%s.png' % i, i)
    random_sample(u'samples/类别采样_%s.png' % i, i)

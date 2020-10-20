"""
Implementation of spectral normalization convolutional layer.
"""
#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf


def spectral_norm(w, iteration=1):
    """
    Implementation of spectral_norm() by Junho Kim at:
       https://github.com/taki0112/Spectral_Normalization-Tensorflow
    """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        # power iteration
        # Usually iteration = 1 will be enough

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def spectral_norm_conv2d(incoming, filters, kernel_size=3, stride=1, padding='same', iteration=1, scope='sn_conv'):
    """
    wrapper to SpectralNormConv2D class
    """
    conv_norm = SpectralNormConv2D(filters, kernel_size, stride, padding, iteration, scope)
    return conv_norm.call(incoming)


class SpectralNormConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, padding='same', iteration=1, scope='sn_conv'):
        """
        Convolutional layer containing a wrapper to spectral_norm()
        :param filters: (int) number of filters
        :param kernel_size: (int) kernel size for the convolutional layer
        :param stride: (int) stride for the convolutional layer
        :param padding: (str) padding to apply to the convolved tensor
        :param iteration: (int) power iteration. Usually iteration = 1 will be enough.
        :param scope: (str) add variable scope
        """
        super(SpectralNormConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding.upper()
        self.iteration = iteration
        self.scope = scope

    def call(self, inputs, **kwargs):
        """ call to the layer
        :param inputs: incoming tensor
        :return:
        """
        with tf.variable_scope(self.scope):
            w = tf.get_variable("kernel", shape=[self.kernel_size, self.kernel_size, inputs.get_shape()[-1], self.filters])
            b = tf.get_variable("bias", [self.filters], initializer=tf.constant_initializer(0.0))

            conv_norm = tf.nn.conv2d(input=inputs, filter=spectral_norm(w, self.iteration),
                                     strides=[1, self.stride, self.stride, 1], padding=self.padding) + b

        return conv_norm

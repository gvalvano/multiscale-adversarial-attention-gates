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
from tensorflow import layers

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class ModalityEncoder(object):

    def __init__(self, input_data, encoded_anatomy, n_latent, is_training, name='ModalityEncoder'):
        """
        Class for the modality encoder architecture.
        :param input_data: (tensor) incoming tensor with input images
        :param encoded_anatomy: (tensor) incoming tensor with input anatomy information
        :param n_latent: (int) number of latent variables for the network output.
        :param is_training: (tf.placeholder(dtype=tf.bool) or bool) variable to define training or test mode; it is
                        needed for the behaviour of dropout, batch normalization, ecc. (which behave differently
                        at train or test time)
        :param name: (string) variable scope for the encoder

        - - - - - - - - - - - - - - - -
        Notice that:
          - the network outputs the parameters of a gaussian distribution (mean and log(variance)) together with a
            data point sampled from such distribution.
        - - - - - - - - - - - - - - - -

        Example of usage:

            # build the entire model:
            model = ModalityEncoder(input_data, encoded_anatomy, n_latent, is_training).build()

            # get mean and log variance for kl divergence loss term:
            z_mean, z_logvar = model.get_z_stats()
            loss = kl_loss(z_mean, z_logvar)

            # rebuild the input as in standard VAE:
            z_sampled = model.get_z_sample()
            decoded_data = decoder(z_sampled)

        """
        # check for compatible input dimensions
        shape = input_data.get_shape().as_list()
        assert not shape[1] % 16
        assert not shape[2] % 16
        assert shape[1:3] == encoded_anatomy.get_shape().as_list()[1:3]  # image width and height must match

        self.input_data = input_data
        self.encoded_anatomy = encoded_anatomy
        self.n_latent = n_latent
        self.is_training = is_training
        self.name = name

        self.z_mean = None
        self.z_logvar = None
        self.z_sampled = None

    def build(self, reuse=tf.AUTO_REUSE):
        """
        Build the model and define:
          - estimate of mean and variance of predicted gaussian distribution
          - sample from the distribution
        :param reuse: (bool) if True, reuse weights.
        :return: self
        """
        with tf.variable_scope(self.name, reuse=reuse):
            incoming = tf.concat([self.input_data, self.encoded_anatomy], axis=-1)

            encoded = self._encode_brick(incoming, self.is_training, scope='encode_brick_0')
            encoded = self._encode_brick(encoded, self.is_training, scope='encode_brick_1')
            encoded = self._encode_brick(encoded, self.is_training, scope='encode_brick_2')
            encoded = self._encode_brick(encoded, self.is_training, scope='encode_brick_3')

            encoded = self._dense_brick(encoded, self.is_training)
            self.z_mean, self.z_logvar, self.z_sampled = self._z_distribution(encoded, self.n_latent)

        return self

    def estimate_z(self, incoming, reuse=True):
        """
        Wrapper to self.build() to estimate z given the incoming tensor.
        :param incoming: (tensor) incoming tensor
        :param reuse: (bool) if True, reuse trained weights
        :return: estimate of z
        """
        input_data = self.input_data
        self.input_data = incoming

        model = self.build(reuse)
        z_regress, _ = model.get_z_stats()
        self.input_data = input_data

        return z_regress

    def get_z_stats(self):
        return self.z_mean, self.z_logvar

    def get_z_sample(self):
        return self.z_sampled

    @staticmethod
    def _encode_brick(incoming, is_training, scope='encode_brick'):
        """ Encoding brick: conv with stride 2 --> batch normalization --> leaky relu.
        """
        with tf.variable_scope(scope):
            conv = layers.conv2d(incoming, filters=16, kernel_size=3, strides=2, padding='same',
                                 kernel_initializer=he_init, bias_initializer=b_init)
            conv_bn = layers.batch_normalization(conv, training=is_training)
            conv_act = tf.nn.leaky_relu(conv_bn)
        return conv_act

    @staticmethod
    def _dense_brick(incoming, is_training, scope='dense_brick'):
        """ Dense brick: flatten --> fully connected --> batch normalization --> leaky relu.
        """
        with tf.variable_scope(scope):
            flat = layers.flatten(incoming)
            fc = layers.dense(flat, units=32)
            fc_bn = layers.batch_normalization(fc, training=is_training)
            fc_act = tf.nn.leaky_relu(fc_bn)
        return fc_act

    @staticmethod
    def _z_distribution(incoming, n_latent, scope='z_distribution'):
        """ z_distribution: computes mean and log var of the probability distribution
        """
        with tf.variable_scope(scope):
            z_mean = layers.dense(incoming, units=n_latent, name='z_mean', trainable=True)
            z_logvar = layers.dense(incoming, units=n_latent, name='z_logvar', trainable=True)

            # sample from normal distribution:
            eps = tf.random_normal(tf.stack([tf.shape(incoming)[0], n_latent]),
                                   dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')

            sampled_z = z_mean + eps * tf.exp(0.5 * z_logvar)

        return z_mean, z_logvar, sampled_z

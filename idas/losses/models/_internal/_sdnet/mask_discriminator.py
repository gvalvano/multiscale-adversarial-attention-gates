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
from .layers.spectral_norm import spectral_norm_conv2d

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class MaskDiscriminator(object):

    def __init__(self, is_training, n_filters=64, n_blocks=4, out_mode='scalar', name='MaskDiscriminator'):
        """
        Class for building the mask discriminator
        :param is_training: (tf.placeholder(dtype=tf.bool) or bool) variable to define training or test mode; it is
                        needed for the behaviour of dropout (which is different at train or test time)
        :param n_filters: (int) number of filters at the first convolutional layer
        :param n_blocks: (int) number of down-sample blocks
        :param out_mode: (str) output mode: valid entries are ['scalar', 'prob_map']. Defaults to 'scalar'.
                        scalar --> outputs a scalar value as in vanilla GANs
                        prob_map --> outputs a probability map of values (each value is a scalar associated to its given
                            receptive field)
        :param name: (str) name for the variable scope

        - - - - - - - - - - - - - - - -
        Notice that:
          - output is linear (this is meant to be used as LeastSquare-GAN)
        - - - - - - - - - - - - - - - -

        Example of usage:

            # build the discriminator model for real and fake data:
            disc_real = MaskDiscriminator(x_real, [params]).build()
            disc_fake = MaskDiscriminator(x_fake, [params]).build(reuse=True)

            # estimate the output of the discriminator:
            y_real = disc_real.get_prediction()
            y_fake = disc_fake.get_prediction()

            # define loss (according to the LeastSquare-GAN objective, assuming labels 0: fake, 1: real)
            loss_discriminator =  0.5*E[(y_real - 1)^2] + 0.5*E[(y_fake - 0)^2]
            loss_generator = 0.5*E[(y_fake - 1)^2 )]

        """
        assert out_mode in ['scalar', 'prob_map']
        self.out_mode = out_mode

        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.is_training = is_training
        self.name = name

        self.output = None

    def build(self, input_tensor, reuse=tf.AUTO_REUSE):
        """
        Build the model.
        :param input_tensor: (tensor) incoming tensor
        :param reuse: (bool) if True, reuse trained weights
        """

        with tf.variable_scope(self.name, reuse=reuse):

            layer = layers.conv2d(input_tensor, filters=self.n_filters, kernel_size=4, strides=2, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            layer = tf.nn.leaky_relu(layer, alpha=0.2)

            for i in range(self.n_blocks):
                nf = self.n_filters * 2 * (2 ** i)
                stride = 1 if i == self.n_blocks - 1 else 2
                scp = 'sn_conv_{0}'.format(str(i))
                layer = self._conv_and_maybe_downsample_block(layer, n_filters=nf, stride=stride, scope=scp)

            if self.out_mode == 'prob_map':
                # output a 2D probability map:
                self.output = spectral_norm_conv2d(layer, filters=1, kernel_size=4, stride=1, padding='valid',
                                                   scope='sn_conv_out')
            elif self.out_mode == 'scalar':
                # output a scalar value:
                layer = tf.layers.flatten(layer)
                self.output = tf.layers.dense(layer, units=1)

            # final activation: linear
        return self

    def get_prediction(self):
        return self.output

    @staticmethod
    def _conv_and_maybe_downsample_block(incoming, n_filters, stride, scope):
        """
        Applies a spectral-norm convolutional layer using given stride. The output activation is a leaky relu.
        :param incoming: incoming tensor
        :param n_filters: number of filters for the convolutional layer
        :param stride: (int) stride to be used for the convolution. Typical value is stride > 1 (i.e. = 2).
        :param scope: variable scope
        :return: leaky_relu activation of the spectral norm convolutional layer
        """
        with tf.variable_scope(scope):
            n_norm = spectral_norm_conv2d(incoming, filters=n_filters, kernel_size=4, stride=stride, padding='same')
            n_norm_act = tf.nn.leaky_relu(n_norm, alpha=0.2)

        return n_norm_act

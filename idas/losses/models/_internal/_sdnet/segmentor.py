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


class Segmentor(object):

    def __init__(self, input_data, n_classes, is_training, name='Segmentor'):
        """
        Class for the Segmentor architecture. It receives the hard-anatomies as input and outputs the segmentation masks
        :param input_data: (tensor) incoming tensor with input = encoded_anatomy
        :param n_classes: (int) number of classes for the network output.
        :param is_training: (tf.placeholder(dtype=tf.bool) or bool) variable to define training or test mode; it is
                        needed for the behaviour of dropout, batch normalization, ecc. (which behave differently
                        at train and test time)
        :param name: (string) variable scope for the segmentor

        - - - - - - - - - - - - - - - -
        Notice that:
          - the output is linear
        - - - - - - - - - - - - - - - -

        Example of usage:

            # build the entire model:
            model = Segmentor(input_data, encoded_anatomy, n_latent, is_training).build()
            output = model.get_output_mask()
            soft_output = tf.nn.softmax(output)

            loss = weighted_cross_entropy(soft_output, y_true)

        """
        # check for compatible input dimensions

        self.input_data = input_data
        self.n_classes = n_classes
        self.is_training = is_training
        self.name = name

    def build(self):
        """
        Build the model.
        """
        with tf.variable_scope(self.name):

            conv1 = layers.conv2d(self.input_data, filters=64, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            conv1_bn = layers.batch_normalization(conv1, training=self.is_training)
            conv1_act = tf.nn.relu(conv1_bn)

            conv2 = layers.conv2d(conv1_act, filters=64, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            conv2_bn = layers.batch_normalization(conv2, training=self.is_training)
            conv2_act = tf.nn.relu(conv2_bn)

            self.output_mask = layers.conv2d(conv2_act, filters=self.n_classes, kernel_size=1, strides=1,
                                             padding='same', kernel_initializer=he_init, bias_initializer=b_init)

        return self

    def get_output_mask(self):
        return self.output_mask
